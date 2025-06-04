import cv2
import matplotlib
import minimalmodbus
import numba
import jupyter
import pip
import numpy as np
from robodk import robomath as rm

# Problem B - Part (1)
def Bin_Identification(image, Marker_IDs, Bin_Locations):
    # Converting the image to gray scale image
    Gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ArUco Dictionary
    ArUco_Dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    Parameters = cv2.aruco.DetectorParameters()
    Detector = cv2.aruco.ArucoDetector(ArUco_Dictionary, Parameters)

    # Markers Detection
    Corners, IDs, rejected = Detector.detectMarkers(Gray_img)
    if IDs is None:
        print("No ArUco markers present in the input image")
        return {}  # Return empty dict if no markers

    IDs = IDs.flatten()
    print(f"Detected the Marker IDs: {IDs}")

    Valid_Indices = [i for i, id_ in enumerate(IDs) if id_ in Marker_IDs]
    print(f"Valid indices: {Valid_Indices}")
    if len(Valid_Indices) == 0:
        print(f"No valid markers found. Expected IDs: {Marker_IDs}, Detected: {IDs}")
        return {}
    elif len(Valid_Indices) != len(Marker_IDs):
        print(f"Warning: Expected {len(Marker_IDs)} markers, found {len(Valid_Indices)} valid markers")

    # Center of each marker
    Centers = []
    Valid_IDs = []
    for i in Valid_Indices:
        Marker_Corners = Corners[i][0]
        Center_X = np.mean(Marker_Corners[:, 0])
        Center_Y = np.mean(Marker_Corners[:, 1])
        Centers.append([Center_X, Center_Y])
        Valid_IDs.append(IDs[i])

    Centers = np.array(Centers, dtype=float)
    Valid_IDs = np.array(Valid_IDs, dtype=int)

    # Sorting the marker centers by y-coordinate (ascending) and x-coordinate (descending)
    Sort_Values = np.lexsort((-Centers[:, 0], Centers[:, 1]))

    # Defining the bin locations
    Given_Bin_Locations = [
        [125, 200, 225],
        [250, 200, 225],
        [125, -75, 225],
        [250, -75, 225]
    ]

    # Convert bin locations to NumPy array for sorting
    Bin_Locations_Array = np.array(Given_Bin_Locations, dtype=float)

    # Sort bin locations by y-coordinate (descending) and x-coordinate (descending)
    Bin_Sort_Indices = np.lexsort((-Bin_Locations_Array[:, 0], -Bin_Locations_Array[:, 1]))
    Sorted_Bin_Locations = Bin_Locations_Array[Bin_Sort_Indices].tolist()

    # Dictionary Mapping to respective marker IDs
    Markers_To_Locations = {}
    for Index, Sorted_Index in enumerate(Sort_Values):
        Marker_ID = Valid_IDs[Sorted_Index].item()  # Use .item() to get scalar
        Location = Sorted_Bin_Locations[Index]  # Use sorted bin locations
        Markers_To_Locations[Marker_ID] = Location
    
    return Markers_To_Locations

# Problem B - Part (2)

def ArUco_Pose_Estimation(Image, Marker_ID, Marker_Len):
    Cam_Matrix = ("cameraMatrix_RobotiqWristCam_640x480.npy")
    Camera_Matrix = np.load(Cam_Matrix)
    Dist_Coefficients = ("distCoeffs_RobotiqWristCam_640x480.npy")
    Dist_Coeff = np.load(Dist_Coefficients)
    
    # ArUco Markers Detection
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect markers
    Corners, IDs, _ = detector.detectMarkers(Image)

    # Marker with ID = 7
    Pose_Mat = None
    for i, marker_ID in enumerate (IDs):
        if marker_ID == Marker_ID:
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(Corners[i], Marker_Len, Camera_Matrix, Dist_Coeff)
            rvec = rvec[0]
            tvec = tvec[0]

            Rotation_Mat,_ = cv2.Rodrigues(rvec)
            Pose_Mat = np.eye(4)
            Pose_Mat[:3, :3] = Rotation_Mat
            Pose_Mat[:3, 3] = tvec
    
    return Pose_Mat


def main():
    print("Problem B - Part (1)")
    Image_Path = "top_of_bins.jpg"
    image = cv2.imread(Image_Path)

    Given_Marker_IDs = [0, 1, 2, 3]
    Given_Bin_Locations = np.array([
        [250, 200, 225],
        [250, -75, 225],
        [125, 200, 225],
        [125, -75, 225]
    ])

    Output = Bin_Identification(image = image, Marker_IDs = Given_Marker_IDs, Bin_Locations = Given_Bin_Locations)

    # Output displaying
    print("Marker IDs with corresponding Bin Locations are:")
    for Marker_ID, Location in Output.items():
        print(f"Marker ID: {Marker_ID} : {Location}")
    
    print("\nProblem B - Part (2)")
    Image_Path_B_2 = "bin_reference_aruco.jpg"
    Image_B_2 = cv2.imread(Image_Path_B_2)

    Marker_ID = 7
    Marker_Len = 50 # in mm
    Marker_Pose = ArUco_Pose_Estimation(Image = Image_B_2, Marker_ID= Marker_ID, Marker_Len= Marker_Len)
    print("\nReference Marker Pose Matrix is:")
    print(Marker_Pose)

    # Converting to array compatible for RoboDK
    Reference_Pose_Mat = rm.Mat(Marker_Pose.tolist())

    # Translation component 
    ArUco_Correct = rm.invH(Reference_Pose_Mat.translationPose())

    Given_Reference_Pose = rm.Pose(-425.0, -300.0, 400.0, -180.0, 0.0, 180.0)

    # Corrected reference pose
    Reference_Pose_Correct = Given_Reference_Pose * ArUco_Correct

    print("\nCorrected Refernce Pose matrix is:")
    print(Reference_Pose_Correct)

    print("\nPose above all bins are:")
    Marker_IDs = [0, 1, 2, 3]
    Tools = ["Hammer", "Plier", "Screwdriver", "Wrench"]

    for ID in Marker_IDs:
        if ID in Output:
            Rel_Transl = Output[ID]
            Transl_Pose = rm.transl(Rel_Transl[0], Rel_Transl[1], Rel_Transl[2])
            Bin_Pose = Reference_Pose_Correct * Transl_Pose
            print(f"Pose above the bin for Marker ID {ID} (Tool: {Tools[ID]}):")
            print(Bin_Pose)

if __name__ == "__main__":
    main()