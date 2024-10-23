"""
This file modifies the mover.py file distributed by the Unity Robotics Hub.
This script generates motion plans for moving the robot in the Unity environment..
"""

from __future__ import print_function

import rospy

import sys
import copy
import math
import moveit_commander

import moveit_msgs.msg
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, OrientationConstraint, BoundingVolume
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
import geometry_msgs.msg
from geometry_msgs.msg import Quaternion, Pose, PoseStamped
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

from ur3_moveit.srv import MoverService, MoverServiceRequest, MoverServiceResponse

from trac_ik_python.trac_ik import IK

import numpy as np

joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
use_joint_goal = True

ik_solver = IK("world","ee_link")

# Between Melodic and Noetic, the return type of plan() changed. moveit_commander has no __version__ variable, so checking the python version as a proxy
if sys.version_info >= (3, 0):
    def planCompat(plan):
        return plan[1]
else:
    def planCompat(plan):
        return plan
        
        
"""
    Given the start angles of the robot, plan a trajectory that ends at the destination pose.
"""
def plan_trajectory(move_group, destination_pose, start_joint_angles): 
    current_joint_state = JointState()
    current_joint_state.name = joint_names
    current_joint_state.position = start_joint_angles

    moveit_robot_state = RobotState()
    moveit_robot_state.joint_state = current_joint_state
    move_group.set_start_state(moveit_robot_state)

    if use_joint_goal:
        print('destination_pose:', destination_pose, type(destination_pose))
        seed_state = start_joint_angles
        p = destination_pose.position
        r = destination_pose.orientation

        joint_target = ik_solver.get_ik(seed_state,p.x,p.y,p.z,r.x,r.y,r.z,r.w)
        print('joint_target:', joint_target)
        if joint_target is None: return None
        move_group.set_joint_value_target(joint_target)
        print('move_group name:', move_group.get_name())
        plan = move_group.plan()
        if plan[0]:
            return planCompat(plan)
        
        print('*'*60)
        print('failure --> return None')
        print('*'*60)
        return None
    else:
        move_group.set_pose_target(destination_pose)
        return planCompat(move_group.plan())


"""
    Creates a motion plan.
    Gripper behaviour is handled outside of this trajectory planning.
    https://github.com/ros-planning/moveit/blob/master/moveit_commander/src/moveit_commander/move_group.py
"""
def plan_pick_and_place(req):
    print('mover.py/plan_pick_and_place')
    setup_scene()

    response = MoverServiceResponse()

    current_robot_joint_configuration = [
        math.radians(req.joints_input.joint_00),
        math.radians(req.joints_input.joint_01),
        math.radians(req.joints_input.joint_02),
        math.radians(req.joints_input.joint_03),
        math.radians(req.joints_input.joint_04),
        math.radians(req.joints_input.joint_05),
    ]
    
    if len(req.poses):
        print('using poses list')
        
        for i, pose in enumerate(req.poses):
            trajectory = plan_trajectory(move_group, pose, current_robot_joint_configuration)
            move_group.clear_pose_targets()
            if trajectory is None:
                print('*'*60)
                print('manipulation sequence generation failed at pose', i)
                print('*'*60)
                return MoverServiceResponse()

            current_robot_joint_configuration = trajectory.joint_trajectory.points[-1].positions
            response.trajectories.append(trajectory)
    
    else:
        # position gripper directly above target location
        pre_grasp_pose = plan_trajectory(move_group, req.pose1, current_robot_joint_configuration)
        move_group.clear_pose_targets()
        if pre_grasp_pose is None:
            print('*'*60)
            print('manipulation sequence generation failed at pose: approach')
            print('*'*60)
            return MoverServiceResponse()

        previous_ending_joint_angles = pre_grasp_pose.joint_trajectory.points[-1].positions
        response.trajectories.append(pre_grasp_pose)

        # lower gripper
        pose2 = copy.deepcopy(req.pose2)
        grasp_pose = plan_trajectory(move_group, pose2, previous_ending_joint_angles)
        move_group.clear_pose_targets()
        if grasp_pose is None:
            print('*'*60)
            print('manipulation sequence generation failed at pose: lower')
            print('*'*60)
            return MoverServiceResponse()

        previous_ending_joint_angles = grasp_pose.joint_trajectory.points[-1].positions
        response.trajectories.append(grasp_pose)

        # raise gripper
        pick_up_pose = plan_trajectory(move_group, req.pose1, previous_ending_joint_angles)
        move_group.clear_pose_targets()
        if pick_up_pose is None:
            print('*'*60)
            print('manipulation sequence generation failed at pose: raise')
            print('*'*60)
            return MoverServiceResponse()
        
        previous_ending_joint_angles = pick_up_pose.joint_trajectory.points[-1].positions
        response.trajectories.append(pick_up_pose)

    print('*'*60)
    print('successfully generated manipulation sequence')
    print('*'*60)

    move_group.clear_pose_targets()
    return response

def setup_scene():
    global move_group
    
    group_name = "arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    # Add table collider to MoveIt scene
    scene = moveit_commander.PlanningSceneInterface()
    robot = moveit_commander.RobotCommander()
    rospy.sleep(2)
    

def set_pose(poseStamped, pose):
    '''
    pose is an array: [x, y, z]
    '''
    poseStamped.pose.position.x = pose[0]
    poseStamped.pose.position.y = pose[1]
    poseStamped.pose.position.z = pose[2]
    return poseStamped

def moveit_server():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('ur3_moveit_server')

    s = rospy.Service('ur3_moveit', MoverService, plan_pick_and_place)
    print("Ready to plan")

    rospy.spin()


if __name__ == "__main__":
    moveit_server()
