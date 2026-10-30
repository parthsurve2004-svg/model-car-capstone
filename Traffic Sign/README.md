# Pi-only trained Traffic Sign package

This smaller package includes the actual trained FP16 NCNN model, application code, printable signs, and test reports. It omits the duplicate PyTorch checkpoint and sample training photos to reduce download size.

Keep this Traffic Sign folder in your model-car-capstone repository. Install the project dependencies, including ncnn, then use:

```bash
python "Traffic Sign/detect.py" --source "Traffic Sign/printable_cards/stop.png" --output stop-check
python "Traffic Sign/run.py" --mode lane --bench-safe
```

The existing general obstacle and pothole models must remain available in the original repository paths. The NCNN sign model loads automatically. Check stationary real-camera recognition before moving the vehicle. Synthetic training results do not prove physical accuracy or stopping distance. See GUIDE.md and MODEL_CARD.md.

Only the NCNN artifact is included here; references to PyTorch training/evaluation describe the model used to produce and validate this export.
