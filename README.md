# The Eye in the Sky
This is the repository for the problem statement *The Eye in the Sky* for the Inter IIT Tech Meet 2018 at IIT Bombay.
The approach used is: [Interactive Medical Image Segmentation using Deep Learning with Image-specific Fine-tuning](http://discovery.ucl.ac.uk/10032237/7/David_08270673.pdf)

## Instructions
1. Download the dataset.
2. Install the python packages:
  ```
  pip install -r requirements.txt
  ```
3. Run the command:
  ```
  ./dataset_rot.py --data-path /path/to/dataset
  ```
4. Next, run the command: `./main.py -h` to view the necessary arguments.
5. Run `main.py` with the necessary arguments.

## Instruction to reproduce the best model results
1 `python best_model.py --data-path /path/to/data --max-steps 500000 --early-stop-diff 5e-12 --early-stop-steps 0 --save-dir /data/avikpal/model2.ckpt`
