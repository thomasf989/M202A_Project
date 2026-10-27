# Upstream components

Source: https://github.com/cvg/LightGlue
Commit: eb42fee2d71449efb0aa5c10549752b5d75384d8

`lightglue.py` and `utils.py` are unmodified upstream files under Apache-2.0.
`aliked.py` is unmodified and includes its BSD-3-Clause notice.
Only the package initializer was reduced to import the used components.
No SuperPoint implementation or weights are redistributed.
Pretrained ALIKED and LightGlue weights are downloaded from their upstream URLs
on first enhanced localization and cached by PyTorch.
