# Replication gate, declared before coffee test evaluation

The initial DSEC attempt hit severe resource pressure (subsequently traced to
launcher construction order; see RUN_LOG.md). The full coffee seed-0 pair is the first
transfer gate. If the frozen candidate improves 8-bit midpoint PSNR and does not
worsen mean SSIM or VGG LPIPS, run fresh matched coffee pairs at seeds 1 and 2
without changing any model setting. Only three positive PSNR pairs with overall
SSIM/LPIPS improvement support a repeatable gain on this one scene.

If the first gate fails, report the failed native transfer and do not claim a
stable advantage; this is not a proof that every possible seed would lose.
No validation-driven structure or learning-rate change is allowed in this test.
Even passing three seeds would not establish cross-scene stability.
