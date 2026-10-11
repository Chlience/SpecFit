# Docker image

`Dockerfile` builds the image used for all measurements on top of the NGC
PyTorch 25.11 base pinned in `base-image.lock.json`, with the extra packages
pinned in `requirements.overlay.lock`. `base-packages.lock.json` records the
package versions of the base image, and `models.lock.json` pins the revisions
of the target model and the EAGLE-3 drafter.

Build with `bash scripts/docker/build-image.sh` (`BASE_IMAGE` and
`PIP_INDEX_URL` override the defaults). `scripts/docker/run-a100.sh` starts a
container on one GPU with the following mounts:

| Content | Host (environment variable) | Container |
| --- | --- | --- |
| Models | `SPECFIT_MODEL_ROOT_HOST` | `/models` |
| Packed expert checkpoint | `SPECFIT_PACKED_CACHE_ROOT_HOST` | `/packed-cache` |
| Inputs and results | `SPECFIT_DATA_ROOT` | `/data` |
| Source checkout | current checkout | `/workspace/specfit` |

`SPECFIT_GPU` selects the GPU and `SPECFIT_IMAGE` the image. The measurement entry
points record the image ID, the model and checkpoint identities and the source
revision in every manifest; the controllers compare them with the values that
were frozen for the runs in the paper (constants such as `IMAGE` at the top of
the controller scripts), so a new build requires updating those constants.
