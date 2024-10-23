# Recognising Affordances in Predicted Futures to Plan With Consideration of Non-Canonical Affordance Effects

Solvi Arnold(1), Mami Kuroishi(2), Rin Karashima(2), Tadashi Adachi(2) and Kimitoshi Yamazaki(1)

(1) [Shinshu University, Autonomous Intelligence & Systems Lab](http://www.ais.shinshu-u.ac.jp)
(2) [EPSON AVASYS](https://avasys.jp)

## Preliminaries

Unity is required for generating data, creating tasks, and executing generated plans.

We confirmed operation with the following environment/library versions:
```
Unity: 2021.2.9f1
Pytorch: 2.1.2
JAX: 0.4.13
numpy: 1.24.4
cv2: 4.6.0
imageio: 2.9.0
tqdm: 4.49.0
yaml: 6.0
matplotlib 3.3.2
```

Below we explain the full process for generating data in simulation, training and evaluating the affordance recognition module, training and evaluating the affordance effect prediction module, creating planning tasks, generating plans, and executing plans in simulation.

-----
## ROS setup

Open `sim/ROS/src/ur3_moveit/config/params.yaml` and enter your ROS_IP.

Navigate to the `sim/ROS` directory
Run:
>`catkin build`

And wait for everything to build.
Note: A few warnings may be displayed during building. This is expected.

-----

## Data generation

Launch Unity Hub.

Click on `ADD`.

Select the `sim/unity_project` directory included with this repository and click `OK`

The project list should now display `unity_project`. Click on it to load it.

At this point you will be missing requires packages still, so the project does not compile. Select `Ignore` on the warning message.
Next we install the missing packages.
From the `Window` menu, select `Package Manager`.
Click on the + symbol, and select `Add package from git URL...`.
Install the following packages with the provided git URLs:

1. [Perception package](https://github.com/Unity-Technologies/com.unity.perception) - `com.unity.perception@0.8.0-preview.3`
2. [URDF Importer package](https://github.com/Unity-Technologies/URDF-Importer) - `https://github.com/Unity-Technologies/URDF-Importer.git?path=/com.unity.robotics.urdf-importer#v0.2.0-light`
3. [TCP Connector package](https://github.com/Unity-Technologies/ROS-TCP-Connector) - `https://github.com/Unity-Technologies/ROS-TCP-Connector.git?path=/com.unity.robotics.ros-tcp-connector#v0.2.0-light`

For more detailed instructions about installing packages, see: https://github.com/Unity-Technologies/Robotics-Object-Pose-Estimation/blob/main/Documentation/1_set_up_the_scene.md#step-2 .

From `File` --> `Open Scene`, open `sim/unity_project/Assets/Scenes/AffordanceScene.unity`.

If the scene objects are bright pink or invisible, go back to the `Package Manager`, search for `Universal RP` (Universal Render Pipeline) from the Unity Registry, and install it.

Open the `Robotics/ROS Settings` dialog and enter your ROS_IP.

Select the `AffordanceChecker` object in the object list.
With the `AffordanceChecker` object selected, the Inspector panel will show configuration options.

Check `Enable Scene Randomisation`

Check `Bypass Robot Motion`
This skips all unnecessary robot motion, saving a lot of time.
(Generating without bypass is possible but not recommended. If you need to generate data without bypass, you must launch the ROS side to plan robot motions.)

Enter a path to save the generated data to in the `Data Save Dir`.

Press the play button (▶).

Press the `Generate Data` button.

Leave running until you're happy with the amount of data.
Note that under the default settings, the recognition and prediction modules use 500 sequences as validation data, 500 sequences as test data, and the remainder as training data, so you will need >>1000 sequences.
In the paper we used a dataset of 26563 sequences total.

Unpress the play button.

Now you should have a dataset in the directory you specified in the `Data Save Dir` field.
The last sequence may be incomplete, so you may want to manually delete it.

### Recovery when data generation gets stuck
Occasionally, poorly handled object collisions and such may cause data generation to get stuck.
When this happens, note the current value of the `Data Index` field in the Inspector panel for the `AffordanceChecker` object.
Stop the simulation (unpress play). The `Data Index` field will now jump back to its initial value.
set the `Data Index` field to one before the value where the simulation got stuck, and continue data generation from there.

-----

## Training the affordance recognition module

The affordance prediction module is a ScaledYOLOv4 network with a number of modifications.
The original ScaledYOLOv4 net can be found at https://github.com/WongKinYiu/ScaledYOLOv4 .

Navigate to the `/recognition` directory.

Open `preprocess_data.py`.

This script processes the unity dataset into the right format for training the affordance recognition network.

Near the top of the file, specify the following data source and destination paths:
- `src_dirs`: A list of paths to directories from which to read Unity data.
If you have just one dataset, this should be a singleton list.
- `dst_dir`: Path to the directory to save the preprocessed dataset to.

Save `preprocess_data.py` and run it:
>	`python preprocess_data.py`

Preprocessing can take a while.

Once preprocessing is done, start training by running:
>`python train.py --data=mydata/DATASET_NAME/data.yaml --hyp=data/hyp.scratch.yaml --cfg=models/yolov4-csp_aff.yaml --weights=None --noautoanchor --img-size=128`

Where `mydata/DATA_SET_NAME/data.yaml` is the path to the yaml file produced by the data preprocessing script, in the `dst_dir` directory defined above.

Each training run produces a new `expX` directory in `/recognition/runs/` (with X an integer).
During training, network weights and images of intermediate results on test batches are stored to this directory.
Old intermediate network weights are not automatically removed, so you may want to discard those to save space.

-----

## Evaluating the trained affordance recognition module

Navigate to the `/recognition` directory.

Open `evaluation_recognition.py`.
At the top of the file, change the value of `dataset_dir` to the path of the preprocessed dataset.

Run:
>`python evaluate_recognition.py --weights=runs/expX/weights/best.pt --set=SET --conf-thres=THRESHOLD`

Where:
`X`: Integer indicating the training run to use.
`SET`: One of train/validation/test.
`THRESHOLD`: The confidence threshold for accepting or rejecting a recognised affordance.

This will do a few things:
- Save images of ground truths and recognitions to the `/recognition_evaluation` directory.
- Display aggregate evaluation results in the terminal.
- Display the threshold value that would optimise the number of correctly recognised affordances on the selected set.

You can run the evaluation routine on the validation set to determine a suitable threshold value, and then evaluate all sets using that threshold value.

Save the obtained threshold value for use in planning.

-----

## Training the affordance effect prediction module

Navigate to the `/prediction` directory.

Open `data_manager_aff.py`.

Near the top of the file, specify the following paths:

- `utils_path`: Absolute path to `/prediction/utils.py`.
(We import this module from an absolute path to avoid clashing with the utils module of the recognition module during planning.)

- `data_paths`: A list of paths to directories from which to read Unity data.
If you have just one dataset, this should be a singleton list. This should usually be the same as `src_dirs` in the recognition module.

- `preprocessed_data_file_path`: File path to save a single-file archive of preprocessed data.
Data is preprocessed differently for the prediction module and the recognition module, so this is independent from `dst_dir` in the recognition module.
By default the preprocessed data archive is stored to `/prediction/preprocessed_data.npy`.

Start training by running:
>`python pEMD.py RUN_NAME`

Where `RUN_NAME` is a freely chosen name for the run.

If a run by the name `RUN_NAME` already exists, training will be resumed from its final saved state.

If the dataset has not been preprocessed yet, it will be preprocessed before training starts and saved to `preprocessed_data_file_path`.
If the dataset has already been preprocessed, you will be prompted to choose whether to use the existing preprocessed data file, or discard it and preprocess the data again.
If any changes or additions to the dataset have been made, you should choose to reprocess at this point (or manually delete the preprocessed data file).

### Resolving memory errors
If training fails with OOM (Out Of Memory) errors, open `config_aff.py` and reduce `n_batch_generation_processes`.
This setting determines the number of subprocesses that is launched to generate training batches in parallel with the main training process.
Each of these subprocesses claims some GPU memory when it is launched, which can cause you to run out of GPU memory on the main process.
Reducing the number of processes too much will slow down learning unnecessarily, so you may need to tune the value for your hardware.

-----

## Evaluating the trained affordance effect prediction module

Navigate to the `/prediction` directory.

Run:
>`python evaluation_aff.py RUN_NAME --set=SET`

Where:
`RUN_NAME`: The name of a completed training run.
`SET`: One of train/validation/test.

This will do two things:
- Save images of ground truths and predictions to the `/recognition/RUN_NAME/prediction_evaluation` directory.
- Display aggregate evaluation results for the specified set in the terminal.

-----

## Creating a planning task

Launch Unity.

Open the `unity_project` project.

Open the `AffordanceScene` scene.

Enter a path to save the task to in the `Planning Task Dir` field.

Create your task's goal state in the editor.

Press the play button (▶).

Press the `Send Goal` button.

Unpress the play button.

Create your task's initial state in the editor.

Press the play button.

Press the `Send State` button.

Unpress the play button.

Manually write a text file specifying the area of the goal state image to use as goal definition for planning as follows.
Open the directory you specified in the `Planning Task Dir` field.
Open `/goal/unity_state_top_RGB.png`.
Determine the area to use as goal patch.
In the `/goal` directory, create a file named `crop.txt`.
Enter the pixel coordinates specifying goal patch area as follows:
>`top_left_x top_left_y bottom_right_x bottom_right_y`

So the content of your "crop.txt" could for example look like this:
>`216 327 295 350`

Which would specify the rectangular area between pixel coordinates `(216,327)` and `(295,350)` as the goal patch.

-----

## Generating an action plan

Navigate to the `/planning` directory.

The planning script needs to know the location of the recognition and prediction source code.
Open `plan.py`.
Near the top of the file, specify the following paths:
- `recognition_dir`: path to `/recognition`.
- `prediction_dir`: path to `/prediction`.

Run:
>`python plan.py --run_name=PATH_TO_PREDICTION_RUN --weights=PATH_TO_RECOGNITION_NET_WEIGHTS --task=TASK_DIR --conf-thres=THRESHOLD -d=DEPTH [-n] [-c] [-z]`

Where:
`PATH_TO_PREDICTION_RUN`: Path to the directory containing the prediction network to use.
`PATH_TO_RECOGNITION_NET_WEIGHTS`: Path to the weights file of the recognition network to use (.pt file).
`TASK_DIR`: Path to a directory containing the task to solve.
`DEPTH`: Search depth for plan search (integer).
`THRESHOLD`: The threshold value found during evaluation of the affordance recognition module.
`[-n]`: Set this flag to run with negative goal (i.e. the goal is to eliminate close matches to the goal patch).
`[-c]`: Set this flag to ignore the depth channel in the calculation of the planning loss (as used in the negative goals cases in the paper).
`[-z]`: Set this flag to ignore the colour channels in the calculation of the planning loss (i.e. plan using depth channel only). Not compatible with `-c`, for obvious reasons.

Note that the way trained models are specified differs between the recognition and prediction modules.
For the prediction module we specify the directory of the training run we wish to use (`PATH_TO_PREDICTION_RUN`), while for the recognition module we specify the weights file (.pt file) to use (`PATH_TO_RECOGNITION_NET_WEIGHTS`).

Let the planning process run to completion.
You should now have a plan in the `TASK_DIR/solution` directory.
You should find visualisations of the generated plan in this directory, and a file named `plan.txt`.
`plan.txt` contains a sequence of actions that is executable in the Unity environment.

-----

## Executing a generated plan

Launch Unity

Open the `unity_project` project.

Open the `AffordanceScene` scene.

Open the `Robotics/ROS Settings` dialog and enter your ROS_IP (if you had not already).

Select the `AffordanceChecker` object in the object list.

With the `AffordanceChecker` object selected, the Inspector panel will show configuration options.

Uncheck `Enable Scene Randomisation`.

Uncheck `Bypass Robot Motion`.

Enter the path of the task to execute in the `Planning Task Dir` field.

Launch the ROS motion planning service as follows.
Open a terminal.
Navigate to the `sim/ROS` directory
Execute the following commands:
>`source devel/setup.bash`
`roslaunch ur3_moveit pose_est.launch`

Wait for `You can start planning now` to be displayed in the terminal.
Note: It is expected behaviour that a few warnings and errors are displayed when launching.

Press the play button (▶).

Press the `Execute Plan` button.

[Optional] When execution is finished, press the `Capture Outcome` button to capture an image of the outcome to the task directory.
	MAY BE SUPERFLUOUS (each step already saved...)

Unpress the play button.

-----

## Code origin & Licenses

Code in the `sim/unity_project` directory builds on tutorial code from the Unity Robotics Hub (https://github.com/Unity-Technologies/Unity-Robotics-Hub).
Code in this directory is licensed under the Apache 2.0 license UNLESS marked otherwise at either the file or directory level.

Code in the `sim/ROS` directory includes code from the Unity Robotics Hub and various ROS packages.
Code in this directory is licensed under the Apache 2.0 license UNLESS marked otherwise at either the file or directory level.

Code in the `/recognition` directory is modified from ScaledYOLOv4 (https://github.com/WongKinYiu/ScaledYOLOv4).
Code in this directory is licensed under the GNU General Public License Version 3.0 UNLESS marked otherwise at the file level.

Code in the `/prediction` directory is licensed under the MIT License.

Code in the `/planning` directory is licensed under the GNU General Public License Version 3.0 (due to dependence on the recognition module).


-----
## Citation
```
@ARTICLE{10025365,
  author={Arnold, Solvi and Kuroishi, Mami and Karashima, Rin and Adachi, Tadashi and Yamazaki, Kimitoshi},
  journal={IEEE Robotics and Automation Letters}, 
  title={Recognising Affordances in Predicted Futures to Plan With Consideration of Non-Canonical Affordance Effects}, 
  year={2023},
  volume={8},
  number={3},
  pages={1455-1462},
  keywords={Affordances;Planning;Task analysis;Grippers;Robots;Predictive models;Artificial neural networks;Affordances;cognitive control architectures;deep learning methods;manipulation planning;predictive modelling},
  doi={10.1109/LRA.2023.3239308}}
```
