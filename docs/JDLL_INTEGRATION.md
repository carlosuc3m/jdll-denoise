# JDLL/Appose integration example

Appose is owned by the Java caller; this package only sees plain arrays and
callables. A bridge can translate events without importing Appose here:

```python
import numpy as np
from jdll_denoise import DenoisingCancelled, denoise

def run_for_task(shared_pixels, task):
    image = np.asarray(shared_pixels)

    def forward(event):
        task.update(
            event["message"],
            event["current"],
            event["maximum"],
            event,
        )

    result = denoise(
        image,
        {
            "method": "noise2fast",
            "effort": "balanced",
            "device": "auto",
            "dimensions": "auto",
            "axes": "tczyx",
            "seed": 5489,
        },
        callbacks=forward,
        cancellation=lambda: task.cancel_requested,
    )
    # Contiguous float32 is suitable for publishing through shared memory.
    return result["denoised"], result["metadata"]
```

Call `capabilities()` before rendering acceleration or dimension controls.
Blend strength, source-adjacent file writing, host metadata, and output naming
remain Java responsibilities.

## ZS-N2N version-1 contract

Before running ZS-N2N, the Java bridge must check
`capabilities()['methods']['zs_n2n']['effort_presets_version'] >= 1` and that
the selected effort occurs in the advertised `efforts` list. Missing/older
capability data requires an update-required message for ZS-N2N only; other
methods remain available. Do not silently run the former width-48 presets.

Forward `method="zs_n2n"` and the effort ID, using canonical `balanced_high`
for GUI label "Balanced-high". Show the fourth radio option only for ZS-N2N;
switching to another method while it is selected resets effort to Balanced.
Invalidate preview caches when effort changes. Python's `presets.py` owns the
architecture/update table; Java must not copy it or substitute device-specific
settings. Display the resolved initialization message and progress maximum.

Use ZS-N2N's min-max defaults (omit normalization or send low=0/high=100,
per_channel=true/clip=true), not forced 0.1/99.9 clipping. Explicit caller
normalization overrides remain honored by Python. Forward ordinary callback
dictionaries without Appose imports in the backend. These Java-side changes
belong in `/home/carlos/git/JDLL`; no Java files are changed here.

## Noise2Fast effort and progress

Continue sending the effort identifier and device. Balanced remains the default;
the backend's `presets.py` is the only executable preset mapping. Display the
initialization status or read the actual width/patience from `resolved_config`
(including any advanced overrides); do not duplicate the table in Java.
Width controls hidden feature channels. Patience counts consecutive optimizer
steps without a strictly improved full-image stopping score. The output average
always uses the last 100 available predictions, independently of patience.

For optimization events, `maximum=0` means indeterminate duration. Show completed
steps and the stale/patience counter, not a percentage or an estimate based on
patience. A positive maximum is an explicit cap, not a promise to run that many
steps. The ordinary completion event still reports 1/1. The historical metric
`epochs_without_validation_improvement` counts steps. Forward the dictionaries
unchanged; the bridge/UI must handle zero maximum without dividing by it.

## Required Java follow-up (not implemented in this Python repository)

- In `/home/carlos/git/JDLL/src/main/java/io/bioimage/modelrunner/model/special/denoise/DenoisingConfig.java`,
  let Noise2Fast use the backend's per-channel 0/100 percentile (min-max)
  defaults instead of unconditionally sending 0.1/99.9 clipping. Python must
  continue honoring valid explicit normalization overrides.
- In `DenoisingPluginUI.java`, show resolved width/patience and indeterminate
  optimization totals. Keep the existing effort radio buttons and Balanced
  default; forward settings and callbacks without a second preset table.
