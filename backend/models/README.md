# Put the checkpoint here

Copy one of these out of your Kaggle download:

    best_weights.pt    75 MB, weights only - preferred, this is all inference needs
    best.pt           226 MB, full state including optimizer - also works

The server looks for `best_weights.pt` first, then `best.pt`. Override with
the `CKPT_PATH` environment variable.

If your download is named `best_weights (1).pt`, rename it. The space and
parentheses break path handling.

The checkpoint is large and patient-model derived, so `.gitignore` excludes
`*.pt`. Do not commit it.
