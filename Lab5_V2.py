import tensorflow as tf
import keras as keras
import numpy as np
import cv2
from cv2 import aruco as aruco
import time
import minimalmodbus
import numpy as np

from RobotiqGripper import*
from robolink import *
from robodk import *

# IMPORT YOUR OWN CLASSES/FUNCTIONS HERE
from hw2_a_driver import Bin_Identification, ArUco_Pose_Estimation

# ------------ PARAMETERS YOU SHOULDN'T EDIT---------
# CAMERA PARAMETERS
CAMERA_FOCUS = 420.0
CAMERA_INDEX = 0
RESOLUTION = (640,480)

# ARUCO PARAMETERS
MARKER_LENGTH = 50.0 # [mm]
CAMERA_MATRIX_FILE = f'cameraMatrix_RobotiqWristCam_640x480.npy'
DISTORTION_COEFF_FILE = f'distCoeffs_RobotiqWristCam_640x480.npy'
cameraMatrix = np.load(CAMERA_MATRIX_FILE)
distCoeffs = np.load(DISTORTION_COEFF_FILE)
aruco_dict = aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50)  # Choose an ArUco dictionary ID
parameters = cv2.aruco.DetectorParameters()
font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text

#-----------------------------------------------------

# ------- PARAMETERS YOU CAN EDIT AS REQUIRED --------
RUN_ON_ROBOT = False
GRIPPER_PORT = 'COM5'
MODEL_PATH = 'lab5model_apr15.keras'
classes = {0: 'Hammer', 1: 'Pliers', 2: 'Screwdriver', 3: 'Wrench'}

#------------------------------------------------------

#--------------CODE STARTS HERE--------------------------#
RDK = Robolink() 
robot = RDK.ItemUserPick('UR5', ITEM_TYPE_ROBOT)
if not robot.Valid():
    raise Exception('No robot selected or available')

if RUN_ON_ROBOT: 
    success = robot.Connect()
    status, status_msg = robot.ConnectedState()
    if status != ROBOTCOM_READY:
        print(status_msg)
        raise Exception("Failed to connect: " + status_msg)
    RDK.setRunMode(RUNMODE_RUN_ROBOT)

joints_ref = robot.Joints()
target_ref = robot.Pose()
pos_ref = target_ref.Pos()
robot.setPoseFrame(robot.PoseFrame())
robot.setPoseTool(robot.PoseTool())
speed_linear = 200
speed_joint = 150
accel_linear = 200
accel_joints = 360
robot.setSpeed(speed_linear, speed_joint, accel_linear, accel_joints)

if RUN_ON_ROBOT: 
    instrument = minimalmodbus.Instrument(GRIPPER_PORT, 9, debug=False)
    instrument.serial.baudrate = 115200
    gripper = RobotiqGripper(portname=GRIPPER_PORT, slaveaddress=9)
    gripper.activate()

if RUN_ON_ROBOT:
    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        print("Cannot open camera {}".format(CAMERA_INDEX))
        exit()
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cam.set(cv2.CAP_PROP_FOCUS, CAMERA_FOCUS)

def LoadModel(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def Capture_And_Save_Last_Frame(Image_Path, camera_Idx, Frames, Resolution, Cam_Focus):
    Image_Capture = cv2.VideoCapture(camera_Idx)
    if not Image_Capture.isOpened():
        print("Unable to start camera {}".format(camera_Idx))
        exit()

    if RUN_ON_ROBOT:
        Image_Capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        Image_Capture.set(cv2.CAP_PROP_FRAME_WIDTH, Resolution[0])
        Image_Capture.set(cv2.CAP_PROP_FRAME_HEIGHT, Resolution[1])
        Image_Capture.set(cv2.CAP_PROP_FOCUS, Cam_Focus)

    Final_Frame = None
    for i in range(Frames):
        ret, frame = Image_Capture.read()
        if not ret:
            Image_Capture.release()
            return None
        Final_Frame = frame
    Image_Capture.release()

    if Final_Frame is not None:
        cv2.imwrite(Image_Path, Final_Frame)
    return Final_Frame

# HOMOGENEOUS TRANSFORMATION MATRICES
original_tool_pose = transl(0, 0, 190)
flange_to_camera = transl(0, 43.75, 10) * rotx(np.deg2rad(-30))
tcp_to_camera = invH(original_tool_pose) * flange_to_camera
camera_to_tcp = invH(flange_to_camera) * original_tool_pose

tool_relative_x_coordinates = [0, -225, -450, -640]
above_bins_offset = [200, 0, -100]
bin_relative_translations = [
    [250, -75, 225],
    [125, -75, 225],
    [125, 200, 225],
    [250, 200, 225]
]

bin_reference = RDK.Item('bin reference').Pose()
bin_reference2 = [23.122886, -82.230903, 57.897172, -65.666268, -90.000000, -66.877114]
Home = [0.000000, -90.000000, 0.000000, -90.000000, 0.000000, 0.000000]
Tool_0 = RDK.Item('tool 0').Pose()
Tool_1 = RDK.Item('Tool 1').Pose()
Tool_Inter_2 = RDK.Item('Inter Tool 2').Pose()
Tool_2 = RDK.Item('Tool 2').Pose()
Tool_Inter_3 = RDK.Item('Inter Tool 3').Pose()
Tool_3 = RDK.Item('Tool 3').Pose()
Tool_Inter_4 = RDK.Item('Inter Tool 4').Pose()
Tool_4 = RDK.Item('Tool 4').Pose()

Homo_Bin_Reference = bin_reference * camera_to_tcp
Homo_Tool_1 = Tool_0 * camera_to_tcp
Homo_Tool_2 = Tool_Inter_2 * camera_to_tcp
Homo_Tool_3 = Tool_Inter_3 * camera_to_tcp
Homo_Tool_4 = Tool_Inter_4 * camera_to_tcp

robot.MoveJ(bin_reference)
robot.MoveJ(Homo_Bin_Reference)

ArUco_Capture = Capture_And_Save_Last_Frame("Reference_Marker_Img.jpg", CAMERA_INDEX, 10, RESOLUTION, CAMERA_FOCUS)

Img_Reference_ArUco = "Reference_Marker_Img.jpg"
Marker_ID = 7
Pose_Aruco = ArUco_Pose_Estimation(Img_Reference_ArUco, Marker_ID, MARKER_LENGTH)

Dummy_ArUco_Pose_Correction = Mat(np.eye(4).tolist())
Correct_bin_reference = bin_reference * Dummy_ArUco_Pose_Correction
above_bins_pose = Correct_bin_reference * transl(above_bins_offset)
Homo_above_bins_pose = above_bins_pose * camera_to_tcp

robot.MoveL(Homo_above_bins_pose)

All_Bins = Capture_And_Save_Last_Frame("All_Bins.jpg", CAMERA_INDEX, 10, RESOLUTION, CAMERA_FOCUS)

Valid_Marker_IDs = [0, 1, 2, 3]
Img_All_Bins_ArUco = "All_Bins.jpg"
Bin_Orientations = Bin_Identification(Img_All_Bins_ArUco, Valid_Marker_IDs, bin_relative_translations)

Tools = {
    0: "hammer",
    1: "Pliers",
    2: "Screwdriver",
    3: "Wrench"
}

Bin_Poses = {
    Tools[k]: Correct_bin_reference * transl(*offset)
    for k, offset in Bin_Orientations.items()
}

Hammer_In_Bin = Bin_Poses["hammer"]
Pliers_In_Bin = Bin_Poses["Pliers"]
Screwdriver_In_Bin = Bin_Poses["Screwdriver"]
Wrench_In_Bin = Bin_Poses["Wrench"]
print(Bin_Poses)

# Tool 1 Operations
robot.MoveL(Homo_Tool_1)
Tool_1_Image = Capture_And_Save_Last_Frame("Tool_1_Image.jpg", CAMERA_INDEX, 10, RESOLUTION, CAMERA_FOCUS)
if RUN_ON_ROBOT:
    gripper.openGripper(speed=255, force=255)
robot.MoveJ(Tool_0)
robot.MoveL(Tool_1)
if RUN_ON_ROBOT:
    gripper.closeGripper(speed=255, force=255)
robot.MoveL(Tool_0)
robot.MoveL(Wrench_In_Bin)
if RUN_ON_ROBOT:
    gripper.openGripper(speed=255, force=255)

# Tool 2 Operations
robot.MoveL(Tool_Inter_2)
robot.MoveJ(Homo_Tool_2)
Tool_2_Image = Capture_And_Save_Last_Frame("Tool_2_Image.jpg", CAMERA_INDEX, 10, RESOLUTION, CAMERA_FOCUS)
if RUN_ON_ROBOT:
    gripper.openGripper(speed=255, force=255)
robot.MoveJ(Tool_Inter_2)
robot.MoveL(Tool_2)
if RUN_ON_ROBOT:
    gripper.closeGripper(speed=255, force=255)
robot.MoveL(Tool_Inter_2)
robot.MoveL(Screwdriver_In_Bin)
if RUN_ON_ROBOT:
    gripper.openGripper(speed=255, force=255)

# Tool 3 Operations
robot.MoveL(Tool_Inter_3)
robot.MoveJ(Homo_Tool_3)
Tool_3_Image = Capture_And_Save_Last_Frame("Tool_3_Image.jpg", CAMERA_INDEX, 10, RESOLUTION, CAMERA_FOCUS)
if RUN_ON_ROBOT:
    gripper.openGripper(speed=255, force=255)
robot.MoveJ(Tool_Inter_3)
robot.MoveL(Tool_3)
if RUN_ON_ROBOT:
    gripper.closeGripper(speed=255, force=255)
robot.MoveL(Tool_Inter_3)
robot.MoveL(Hammer_In_Bin)
if RUN_ON_ROBOT:
    gripper.openGripper(speed=255, force=255)

# Tool 4 Operations
robot.MoveL(Tool_Inter_4)
robot.MoveJ(Homo_Tool_4)
Tool_4_Image = Capture_And_Save_Last_Frame("Tool_4_Image.jpg", CAMERA_INDEX, 10, RESOLUTION, CAMERA_FOCUS)
if RUN_ON_ROBOT:
    gripper.openGripper(speed=255, force=255)
robot.MoveJ(Tool_Inter_4)
robot.MoveL(Tool_4)
if RUN_ON_ROBOT:
    gripper.closeGripper(speed=255, force=255)
robot.MoveL(Tool_Inter_4)
robot.MoveL(Pliers_In_Bin)
if RUN_ON_ROBOT:
    gripper.openGripper(speed=255, force=255)

robot.MoveJ(Home)