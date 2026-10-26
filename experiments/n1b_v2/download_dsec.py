"""Fetch calibrated real RGB DSEC clips with measured exposure intervals.

Read selected ZIP members using HTTP byte ranges; verify ZIP CRC and save hashes.
No events or synthesized/deblurred images are downloaded.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

BASE='https://download.ifi.uzh.ch/rpg/DSEC/train/'


class RemoteFile(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with self.opener.open(urllib.request.Request(url,method='HEAD'),timeout=30) as r:
            self.size=int(r.headers['Content-Length']);self.etag=r.headers.get('ETag')
        self.pos=0;self.transferred=0

    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,size=-1):
        size=self.size-self.pos if size<0 else min(size,self.size-self.pos)
        if size<=0:return b''
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+size-1}'})
        with self.opener.open(req,timeout=60) as r:
            if r.status!=206:raise RuntimeError('Server must honor byte range')
            data=r.read(size+1)
        if len(data)!=size:raise RuntimeError('Incomplete range')
        self.pos+=size;self.transferred+=size;return data


def fetch_small(url,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url,timeout=30) as r:dest.write_bytes(r.read())
    return {'url':url,'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}


def download(sequence,frames):
    root=Path('dataset/dsec_dynamic')/sequence;root.mkdir(parents=True,exist_ok=True)
    prefix=BASE+sequence+'/'+sequence;manifest={'sequence':sequence,'frames':frames,'files':{},'archives':{}}
    for suffix,name in [('_calibration.zip','calibration.zip'),('_image_timestamps.txt','image_timestamps.txt'),('_image_exposure_timestamps_left.txt','exposure_left.txt'),('_image_exposure_timestamps_right.txt','exposure_right.txt')]:
        manifest['files'][name]=fetch_small(prefix+suffix,root/name)
    with zipfile.ZipFile(root/'calibration.zip') as z:
        for n in ('cam_to_cam.yaml','cam_to_lidar.yaml'):(root/n).write_bytes(z.read(n))
    for side in ('left','right'):
        remote=RemoteFile(prefix+f'_images_rectified_{side}.zip')
        with zipfile.ZipFile(remote) as z:
            infos={index:z.getinfo(f'{index:06d}.png') for index in frames}
            def member(index):
                name=f'{index:06d}.png';p=root/side/name;p.parent.mkdir(exist_ok=True)
                info=infos[index]
                fetched=0
                if not p.exists():
                    # One request per complete local ZIP record, independently
                    # buffered so concurrent reads cannot corrupt a shared seek.
                    end=min(remote.size-1,info.header_offset+30+len(name)+info.compress_size+4096)
                    req=urllib.request.Request(remote.url,headers={'Range':f'bytes={info.header_offset}-{end}'})
                    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    with opener.open(req,timeout=60) as response:
                        if response.status!=206:raise RuntimeError('Range ignored')
                        b=response.read();fetched=len(b)
                    h=struct.unpack_from('<4s5H3L2H',b);assert h[0]==b'PK\x03\x04'
                    begin=30+h[-2]+h[-1];compressed=b[begin:begin+info.compress_size]
                    data=zlib.decompress(compressed,-15) if info.compress_type==8 else compressed
                    if len(data)!=info.file_size or zlib.crc32(data)!=info.CRC:raise RuntimeError('ZIP CRC mismatch')
                    p.write_bytes(data)
                data=p.read_bytes()
                if zlib.crc32(data)!=info.CRC:raise RuntimeError('Existing PNG CRC mismatch')
                return side+'/'+name,{'archive_member':name,'crc32':info.CRC,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()},fetched
            member_bytes=0
            for name,metadata,fetched in ThreadPoolExecutor(max_workers=4).map(member,frames):
                manifest['files'][name]=metadata
                member_bytes+=fetched
        manifest['archives'][side]={'url':remote.url,'size':remote.size,'etag':remote.etag,'central_directory_bytes_read':remote.transferred,'member_bytes_downloaded_this_invocation':member_bytes}
        print(sequence,side,len(frames),'frames',remote.transferred+member_bytes,'bytes transferred this invocation',flush=True)
    # Merge previous members on an incremental fetch.
    path=root/'download_manifest.json'
    if path.exists():
        old=json.loads(path.read_text());old['files'].update(manifest['files']);manifest['files']=old['files'];manifest['frames']=sorted(set(old['frames']+frames))
    path.write_text(json.dumps(manifest,indent=2));return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',default='zurich_city_04_c');p.add_argument('--start',type=int,default=120);p.add_argument('--count',type=int,default=40);p.add_argument('--frames',type=int,nargs='*');a=p.parse_args()
    download(a.sequence,a.frames if a.frames is not None else list(range(a.start,a.start+a.count)))
