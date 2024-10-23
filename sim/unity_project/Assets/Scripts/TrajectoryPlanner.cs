/*
This file modifies the TrajectoryPlanner.cs file distributed by the Unity Robotics Hub.
*/

using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System;
using UnityEngine;
using UnityEngine.UI;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using RosMessageTypes.Ur3Moveit;
using Quaternion = UnityEngine.Quaternion;
using Transform = UnityEngine.Transform;
using Vector3 = UnityEngine.Vector3;


public class TrajectoryPlanner : MonoBehaviour{
    
    // ROS Connector
    private ROSConnection ros;

    // Hardcoded variables 
    private readonly int numRobotJoints = 6;
    private readonly float jointAssignmentWait = 0.02f;
    private readonly float poseAssignmentWait = 0.5f;
    private readonly float gripperAngle = 40f;
    
    // Offsets for gripper poses
    private readonly Vector3 pickApproachPoseOffset = new Vector3(0, 0.67f+0.15f+0.293f+0.02f, 0);
    private readonly Vector3 pickLowerPoseOffset = new Vector3(0, 0.25f, 0);
    private readonly Vector3 placeApproachPoseOffset = new Vector3(0, 0.67f+0.15f+0.293f+0.02f, 0);
    private readonly Vector3 placeLowerPoseOffset = new Vector3(0, 0.01f, 0);
    
    private readonly float turnLift = 0.005f;
    
    // Multipliers correspond to the URDF mimic tag for each joint
    private float[] multipliers = new float[] { 1f, 1f, -1f, -1f, 1f, -1f };
    
    // Orientation is hardcoded with the gripper directly above the object
    private readonly Quaternion pickOrientation = new Quaternion(-0.5f,-0.5f,0.5f,-0.5f);
    
    private readonly float[] tiltPoseRotation = new float[] {0.0f,0.0f,0.0f,0.0f,90.0f,0.0f};
    private readonly float[] capturePoseRotation = new float[] {0.0f,0.0f,0.0f,0.0f,90.0f,0.0f};
    
    // Variables required for ROS communication
    public string rosServiceName = "ur3_moveit";
    private const int isBigEndian = 0;
    private const int step = 4;

    public GameObject robot;
    public Transform goal;
    private GameObject physicsDouble;
    
    // Tracks whether we the gripper is currently holding an object
    [HideInInspector] public bool gripperHoldingObject = false;
    
    [HideInInspector] public bool executionAttempted = false;
    
    // Keeps track of the gripper angle
    private int gripperState = 0;
    
    // Handle on the finger pads for fine grasp control
    private List<GameObject> fingerPads;
    private List<GameObject> ghostFingerPads;

    // Articulation Bodies
    private ArticulationBody[] jointArticulationBodies;
    
    ArticulationBody[] articulationChain;
    private List<ArticulationBody> gripperJoints;
    private RenderTexture renderTexture;
    private GameObject gripperBase;
    private GameObject pickTargetObject;
    Quaternion targetRotQ;
    private MoverServiceResponse rosResponse;
    private bool motionExecutionSuccess;
    private FixedJoint graspJoint;
    private GameObject gripperFollowTarget;
    private float massMemory;
    private string currentAffType;
    private StickyScript[] stickies;
    private List<StickyScript> detachedStickies = new List<StickyScript>();
    private float[] freeParams;
    public float placementObjectOffsetY;
    private AffordanceCheck affordanceCheck;
    private System.Random randomiser = new System.Random();
    
    
    public float FindHeight(GameObject obj){
        float[] d = FindHeightAndRange(obj);
        return d[0];
    }
    
    
    public float[] FindHeightAndRange(GameObject obj){
        Mesh mesh = null;
        MeshFilter mF = obj.GetComponent<MeshFilter>();
        if (mF != null){
            mesh = mF.mesh;
        }
        Vector3[] vertices = mesh.vertices;
        float minY = float.MaxValue;
        float maxY = float.MinValue;
        for (int i = 1; i < vertices.Length; i++){
            Vector3 V = obj.transform.TransformPoint(vertices[i]);
            if (V[1] < minY){
                minY = V[1];
            }
            if (V[1] > maxY){
                maxY = V[1];
            }
        }
        return new float[] {maxY-minY, minY, maxY};
    }
    
    
    private void ToggleCollisionIgnore(bool state){
        
        Collider c = pickTargetObject.GetComponent<Collider>();
        Physics.IgnoreCollision(c,fingerPads[0].GetComponent<Collider>(),state);
        Physics.IgnoreCollision(c,fingerPads[1].GetComponent<Collider>(),state);
        Physics.IgnoreCollision(c,ghostFingerPads[0].GetComponent<Collider>(),state);
        Physics.IgnoreCollision(c,ghostFingerPads[1].GetComponent<Collider>(),state);
        
        // get specific collider types because mesh collider objects can an additional collider...
        c = pickTargetObject.GetComponent<MeshCollider>();
        if (c!=null){
            Physics.IgnoreCollision(c,fingerPads[0].GetComponent<Collider>(),state);
            Physics.IgnoreCollision(c,fingerPads[1].GetComponent<Collider>(),state);
            Physics.IgnoreCollision(c,ghostFingerPads[0].GetComponent<Collider>(),state);
            Physics.IgnoreCollision(c,ghostFingerPads[1].GetComponent<Collider>(),state);
        }
        c = pickTargetObject.GetComponent<BoxCollider>();
        if (c!=null){
            Physics.IgnoreCollision(c,fingerPads[0].GetComponent<Collider>(),state);
            Physics.IgnoreCollision(c,fingerPads[1].GetComponent<Collider>(),state);
            Physics.IgnoreCollision(c,ghostFingerPads[0].GetComponent<Collider>(),state);
            Physics.IgnoreCollision(c,ghostFingerPads[1].GetComponent<Collider>(),state);
        }
        c = pickTargetObject.GetComponent<SphereCollider>();
        if (c!=null){
            Physics.IgnoreCollision(c,fingerPads[0].GetComponent<Collider>(),state);
            Physics.IgnoreCollision(c,fingerPads[1].GetComponent<Collider>(),state);
            Physics.IgnoreCollision(c,ghostFingerPads[0].GetComponent<Collider>(),state);
            Physics.IgnoreCollision(c,ghostFingerPads[1].GetComponent<Collider>(),state);
        }
    }
    

    public IEnumerator IterateToGrip(bool toClose, Action<GameObject> contactCallback){

        int n_steps = toClose ? 60 : gripperState;
        var grippingAngle = gripperAngle * n_steps / 60;
        
        foreach (GameObject pad in ghostFingerPads){
            pad.GetComponent<FingerPad>().collision_started = false;
            pad.GetComponent<FingerPad>().collision_ended = false;
        }
        
        bool idle = false;
        bool grasped = false;
        bool released = false;
        
        for (int step = 1; step < n_steps+1; step++){
        
            if (toClose){
                int n_collisions_started = 0;
                foreach (GameObject pad in ghostFingerPads){
                    if (pad.GetComponent<FingerPad>().collision_started){
                        n_collisions_started += 1;
                    }
                }
                if (n_collisions_started > 1){
                    
                    idle = true;
                    Collider cc = ghostFingerPads[0].GetComponent<FingerPad>().currentCollider;
                    
                    if (pickTargetObject == null){ // target undefined -> set touched object as target (happens when executing externally planned aff)
                        pickTargetObject = cc.gameObject;
                        Debug.Log("Grasp established with unspecified target. Setting pickTargetObject to "+pickTargetObject);
                        Debug.Break();
                    }
                    
                    if (cc.gameObject == pickTargetObject){
                        Debug.Log("Establishing gripper-object link");
                        
                        if (!affordanceCheck.bypassRobotMotion){
                            float[] dd = FindHeightAndRange(pickTargetObject);
                            Debug.Log("grasped object height minY maxY: "+dd[0]+" "+dd[1]+" "+dd[2]);
                            placementObjectOffsetY = gripperBase.transform.position.y-dd[1];
                            Debug.Log("placementObjectOffsetY: "+placementObjectOffsetY);
                        }
                            
                        if (contactCallback != null) contactCallback(pickTargetObject);
                        ToggleCollisionIgnore(true);
                        
                        if (!affordanceCheck.bypassRobotMotion){
                            gripperFollowTarget.transform.SetParent(null);
                            gripperFollowTarget.transform.position = pickTargetObject.transform.position;
                            gripperFollowTarget.transform.rotation = pickTargetObject.transform.rotation;
                            pickTargetObject.transform.SetParent(gripperFollowTarget.transform);
                            gripperFollowTarget.transform.SetParent(gripperBase.transform);
                            GripperLink link = pickTargetObject.AddComponent<GripperLink>();
                            link.setup();
                        }
                        gripperState = step;
                        grasped = true;
                    }
                    else{
                        Debug.Log("gripper collided with wrong object: "+cc.gameObject+" instead of "+pickTargetObject);
                        motionExecutionSuccess = false;
                    }
                    yield break;
                }
                else {
                    idle = false;
                }
            }
            else {
                if (step == Math.Min(gripperState,22)){
                    released = ReleaseGraspedObject();
                }
            }
            
            for (int i = 0; i < gripperJoints.Count; i++)
            {
                var curXDrive = gripperJoints[i].xDrive;
                float r = ((float)step/n_steps);
                if (!toClose){
                    r = 1f-r;
                }
                if (idle){
                    curXDrive.target = 0f;
                    gripperJoints[i].xDrive = curXDrive;
                }
                else{
                    curXDrive.target = multipliers[i] * grippingAngle * r;
                    gripperJoints[i].xDrive = curXDrive;
                }
            }
            yield return new WaitForSeconds(jointAssignmentWait);
        }
        yield return new WaitForSeconds(jointAssignmentWait);
        
        if (toClose && !grasped){
            motionExecutionSuccess = false;
        }
        if (!toClose && !released){
            released = ReleaseGraspedObject();
            if (!released) motionExecutionSuccess = false;
        }
    }
    
    
    private void DetachStickies(){
        foreach (StickyScript sticky in stickies){
            if (sticky.stickingTo == pickTargetObject){
                sticky.Detach();
                detachedStickies.Add(sticky);
            }
        }
        
        for (int i = 0; i < pickTargetObject.transform.childCount; i++) {
            Transform child = pickTargetObject.transform.GetChild(i);
            StickyScript sticky = child.GetComponent<StickyScript>();
            if (sticky != null){
                sticky.Detach();
                detachedStickies.Add(sticky);
            }
        }
    }
    
    
    private void RestoreStickies(){
        foreach (StickyScript sticky in detachedStickies){
            sticky.Forget();
        }
    }
    
    
    public IEnumerator ReleaseGraspedObjectRoutine(){
        while(pickTargetObject.GetComponent<FixedJoint>()!=null){
            Destroy(pickTargetObject.GetComponent<FixedJoint>());
            yield return new WaitForSeconds(poseAssignmentWait);
        }
    }
    
    
    private bool ReleaseGraspedObject(){
        Debug.Log("Releasing gripper-object link");
        ToggleCollisionIgnore(false);
        pickTargetObject.transform.SetParent(null);
        Destroy(pickTargetObject.GetComponent<GripperLink>());
        Rigidbody rb = pickTargetObject.GetComponent<Rigidbody>();
        rb.useGravity = true;
        rb.isKinematic = false;
        return true;
    }
    
    
    public void OnCollisionEnter(Collision collision){
        Debug.Log("bridged OnCollisionEnter");
    }
    
    
    public void OnTriggerEnter(Collider other){
        Debug.Log("bridged OnTriggerEnter");
    }
    

    public void InitializeRobotPose(bool tilt,Action<bool> callback){
        StartCoroutine(InitializeRobotPoseRoutine(tilt, callback));
    }

    
    public IEnumerator InitializeRobotPoseRoutine(bool tilt, Action<bool> callback){
        yield return InitializeRobotPoseRoutine(tilt);
        if (callback != null) callback(true);
    }
    
    
    public IEnumerator InitializeRobotPoseRoutine(bool tilt){
        if (graspJoint!=null){
            Rigidbody rb = graspJoint.gameObject.GetComponent<Rigidbody>();
            massMemory = rb.mass;
            rb.mass=0.00001f;
        }
        
        bool isRotationFinished = false;
        
        var rotationSpeed = 270f;
        float[] rotationSpeeds = new float[numRobotJoints+1];
        for (int i = 1; i < numRobotJoints + 1; i++) rotationSpeeds[i] = rotationSpeed;
        
        while (!isRotationFinished){
            Debug.Log("iterating towards default pose...");
            isRotationFinished = ResetRobotToDefaultPosition(rotationSpeeds,tilt);
            Debug.Log("received isRotationFinished = "+isRotationFinished);
            yield return new WaitForSeconds(jointAssignmentWait);
            Debug.Log("wait ends");
        }
        Debug.Log("InitializeRobotPoseRoutine returns");
    }

    
    private bool ResetRobotToDefaultPosition(float[] rotationSpeeds, bool tilt){
        bool isRotationFinished = true;
        float errorMargin = 0.00001f;
        
        float[] poseRotation = tilt ? tiltPoseRotation : capturePoseRotation;
        
        for (int i = 1; i < numRobotJoints + 1; i++){
            var tempXDrive = articulationChain[i].xDrive;
            float currentRotation = tempXDrive.target;
            
            float rotationChange = rotationSpeeds[i] * Time.fixedDeltaTime;
            
            if (currentRotation > poseRotation[i-1]) rotationChange *= -1;
            
            float d = Mathf.Abs(currentRotation-poseRotation[i-1]);
            if (d < errorMargin) rotationChange = 0;
            else isRotationFinished = false;
            
            if (d < rotationChange) rotationSpeeds[i] /= 2;
            
            // the new xDrive target is the currentRotation summed with the desired change
            float rotationGoal = currentRotation + rotationChange;
            tempXDrive.target = rotationGoal;
            articulationChain[i].xDrive = tempXDrive;
        }
        Debug.Log("ResetRobotToDefaultPosition returns "+isRotationFinished);
        return isRotationFinished;
    }
    

    UR3MoveitJoints CurrentJointConfig(){
        UR3MoveitJoints joints = new UR3MoveitJoints();
        
        joints.joint_00 = jointArticulationBodies[0].xDrive.target;
        joints.joint_01 = jointArticulationBodies[1].xDrive.target;
        joints.joint_02 = jointArticulationBodies[2].xDrive.target;
        joints.joint_03 = jointArticulationBodies[3].xDrive.target;
        joints.joint_04 = jointArticulationBodies[4].xDrive.target;
        joints.joint_05 = jointArticulationBodies[5].xDrive.target;

        return joints;
    }
    
    // Short random motion to shake off any objects supported by this object.
    // Used for data generation by robot motion bypass to emulate the unpredictable effect of lifting objects with other objects on top.
    public IEnumerator launchSupported(GameObject obj){
        Vector3 initPos = obj.transform.position;
        Quaternion initRotQ = obj.transform.rotation;
        Vector3 initRot = initRotQ.eulerAngles;
        
        // make object move straight upward some distance to mimic initial upward lift motion
        Vector3 liftPos = new Vector3(initPos.x,initPos.y+0.4f,initPos.z);
        MoveToTarget m = obj.AddComponent<MoveToTarget>();
        m.SetTarget(liftPos,initRotQ);
        
        // wait for object to reach target (MoveToTarget component self-destructs when target is reached)
        while (true){
            if (obj.GetComponent<MoveToTarget>() == null) break;
            yield return null;
        }
        
        float velRange = 1.0f;
        float angRange = 180.0f;
        StabilityCheckTag[] stabilityCheckTags = FindObjectsOfType<StabilityCheckTag>();
        foreach (StabilityCheckTag tag in stabilityCheckTags){
            if (tag.gameObject==obj)continue;
            Vector3 p = tag.gameObject.transform.position;
            if (Vector3.Distance(p,obj.transform.position)<0.3 && p.y>obj.transform.position.y){
                Rigidbody rb2 = tag.gameObject.GetComponent<Rigidbody>();
                rb2.velocity = new Vector3(velRange*(2f*((float)randomiser.NextDouble()-0.5f)),
                                           velRange*(2f*(float)randomiser.NextDouble()),
                                           velRange*(2f*((float)randomiser.NextDouble()-0.5f)));
                rb2.angularVelocity = new Vector3(angRange*(2f*((float)randomiser.NextDouble()-0.5f)),
                                                  angRange*(2f*((float)randomiser.NextDouble()-0.5f)),
                                                  angRange*(2f*((float)randomiser.NextDouble()-0.5f)));
            }
        }
        
        obj.transform.position = initPos;
        obj.transform.rotation = initRotQ;
        Rigidbody rb = obj.GetComponent<Rigidbody>();
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        rb.Sleep();
    }
    
    public IEnumerator PickAndHold(Vector3 targetPos, Vector3 targetRot, GameObject targetObject, Action<bool> finishedCallback){
        yield return PickAndHold(targetPos, targetRot, targetObject, finishedCallback, null);
    }
    
    public IEnumerator PickAndHold(Vector3 targetPos, Vector3 targetRot, GameObject targetObject, Action<bool> finishedCallback, Action<GameObject> contactCallback){
        Debug.Log("Grasp @ pose: "+targetPos.ToString("F4")+" / "+targetRot.ToString("F4"));
        
        currentAffType = "grasp";
        executionAttempted = false;
        targetRotQ = Quaternion.Euler(targetRot.x, targetRot.y, targetRot.z);
        pickTargetObject = targetObject;
        
        // Make a request object for ROS-side planning
        MoverServiceRequest request = new MoverServiceRequest();
        request.joints_input = CurrentJointConfig();
        
        Vector3 pickPos = targetPos + pickLowerPoseOffset;
        
        if (affordanceCheck.bypassRobotMotion){
            float[] dd = FindHeightAndRange(pickTargetObject);
            placementObjectOffsetY = pickPos.y-dd[1];
            
            Rigidbody rb = pickTargetObject.GetComponent<Rigidbody>();
            rb.useGravity = false;
            yield return launchSupported(pickTargetObject);
            
            contactCallback(pickTargetObject);
            gripperFollowTarget.transform.SetParent(null);
            gripperFollowTarget.transform.position = pickPos;
            gripperFollowTarget.transform.rotation = targetRotQ;
            pickTargetObject.transform.SetParent(gripperFollowTarget.transform);
            gripperFollowTarget.transform.SetParent(gripperBase.transform);
            gripperFollowTarget.transform.localPosition = Vector3.zero;
            gripperFollowTarget.transform.localRotation = Quaternion.Euler(180f,0f,0f);
        
            yield return IterateToGrip(true, null);

            gripperHoldingObject = true;
            motionExecutionSuccess = true;
            // Report success or failure via the callback function
            if (finishedCallback != null) finishedCallback(motionExecutionSuccess);
            yield break;
        }
        
        // Change to tilt stance for easier planning
        yield return StartCoroutine(InitializeRobotPoseRoutine(true));
        
        Vector3 approach = new Vector3(targetPos.x, pickApproachPoseOffset.y, targetPos.z);
        
        request.poses = new RosMessageTypes.Geometry.Pose[]{
            // approach pose
            new RosMessageTypes.Geometry.Pose{ 
                position = approach.To<FLU>(),
                orientation = Quaternion.Euler(90, targetRotQ.eulerAngles.y, 0).To<FLU>()
            },
            // pick pose
            new RosMessageTypes.Geometry.Pose{
                position = pickPos.To<FLU>(),
                orientation = Quaternion.Euler(90, targetRotQ.eulerAngles.y, 0).To<FLU>()
            },
            // retreat pose
            new RosMessageTypes.Geometry.Pose{ 
                position = approach.To<FLU>(),
                orientation = Quaternion.Euler(90, targetRotQ.eulerAngles.y, 0).To<FLU>()
            },
        };
        
        // Send the request and process the response
        yield return HandleRequestAndResponse(request,contactCallback);
        
        // Report success or failure via the callback function
        if (finishedCallback != null) finishedCallback(motionExecutionSuccess);
    }
    
    public IEnumerator PlaceHeldObject(Vector3 targetPos, Vector3 targetRot, Action<bool> finishedCallback){
        Debug.Log("Place @ pose: "+targetPos.ToString("F4")+" / "+targetRot.ToString("F4"));
        Debug.Log("pickTargetObject: "+pickTargetObject);
        
        currentAffType = "place";
        
        executionAttempted = false;
        
        Quaternion targetRotQ = Quaternion.Euler(targetRot.x, targetRot.y, targetRot.z);
        
        Vector3 offset2 = new Vector3(placeLowerPoseOffset.x,placeLowerPoseOffset.y+placementObjectOffsetY,placeLowerPoseOffset.z);
        Vector3 placePos = targetPos + offset2;
        
        if (affordanceCheck.bypassRobotMotion){
            
            gripperFollowTarget.transform.SetParent(null);
            gripperFollowTarget.transform.position = placePos;
            gripperFollowTarget.transform.rotation = targetRotQ;//Quaternion.Euler(0,targetRot.y,0);
            //Debug.Break();
            
            Debug.Log("start IterateToGrip");
            yield return IterateToGrip(false, null);
            Debug.Log("end IterateToGrip");
            
            
            gripperHoldingObject = false;
            motionExecutionSuccess = true;
            // Report success or failure via the callback function
            if (finishedCallback != null) finishedCallback(motionExecutionSuccess);
            yield break;
        }
        
        // Change to tilt stance for easier planning
        yield return StartCoroutine(InitializeRobotPoseRoutine(true));
        
        // Make a request object for ROS-side planning
        MoverServiceRequest request = new MoverServiceRequest();
        request.joints_input = CurrentJointConfig();
        
        Vector3 approach = new Vector3(targetPos.x, placeApproachPoseOffset.y, targetPos.z);
        
        request.poses = new RosMessageTypes.Geometry.Pose[]{
            // approach pose
            new RosMessageTypes.Geometry.Pose{ 
                position = approach.To<FLU>(),
                orientation = Quaternion.Euler(90, targetRotQ.eulerAngles.y, 0).To<FLU>()
            },
            // placement pose
            new RosMessageTypes.Geometry.Pose{
                position = placePos.To<FLU>(),
                orientation = Quaternion.Euler(90, targetRotQ.eulerAngles.y, 0).To<FLU>()
            },
            // retreat pose
            new RosMessageTypes.Geometry.Pose{ 
                position = approach.To<FLU>(),
                orientation = Quaternion.Euler(90, targetRotQ.eulerAngles.y, 0).To<FLU>()
            }};
        
        // Send the request and process the response
        yield return HandleRequestAndResponse(request,null);
        
        // Report success or failure via the callback function
        if (finishedCallback != null) finishedCallback(motionExecutionSuccess);
    }
    
    public IEnumerator TurnObject(Vector3 targetPos, Vector3 targetRot, float turnAngle, GameObject targetObject, Action<bool> finishedCallback){
        Debug.Log("Turn @ pose: "+targetPos.ToString("F4")+" / "+targetRot.ToString("F4"));
        
        currentAffType = "turn";
        
        executionAttempted = false;
        
        targetRotQ = Quaternion.Euler(targetRot.x, targetRot.y, targetRot.z);
        pickTargetObject = targetObject;
        
        if (affordanceCheck.bypassRobotMotion){
            Vector3 objRot = targetObject.transform.rotation.eulerAngles;
            Quaternion finalRotQ = Quaternion.Euler(objRot.x, objRot.y+turnAngle, objRot.z);
            MoveToTarget m = targetObject.AddComponent<MoveToTarget>();
            Vector3 liftPos = targetObject.transform.position;
            liftPos = new Vector3(liftPos.x,liftPos.y+turnLift,liftPos.z);
            m.SetTarget(liftPos,targetObject.transform.rotation);
            m.AddTarget(liftPos,finalRotQ);
            m.AddTarget(targetObject.transform.position,finalRotQ);
            while (true){
                if (targetObject.GetComponent<MoveToTarget>() == null) break;
                yield return null;
            }
            
            gripperHoldingObject = false;
            motionExecutionSuccess = true;
            // Report success or failure via the callback function
            if (finishedCallback != null) finishedCallback(motionExecutionSuccess);
            yield break;
        }
        
        // Change to tilt stance for easier planning
        yield return StartCoroutine(InitializeRobotPoseRoutine(true));
        
        // Make a request object for ROS-side planning
        MoverServiceRequest request = new MoverServiceRequest();
        request.joints_input = CurrentJointConfig();
        
        Vector3 approach = new Vector3(targetPos.x, pickApproachPoseOffset.y, targetPos.z);
        Vector3 liftOffset = new Vector3(pickLowerPoseOffset.x, pickLowerPoseOffset.y+turnLift, pickLowerPoseOffset.z);
        
        request.poses = new RosMessageTypes.Geometry.Pose[]{
            // approach pose
            new RosMessageTypes.Geometry.Pose{ 
                position = approach.To<FLU>(),
                orientation = Quaternion.Euler(90, targetRotQ.eulerAngles.y, 0).To<FLU>()
            },
            // clench pose
            new RosMessageTypes.Geometry.Pose{
                position = (targetPos + pickLowerPoseOffset).To<FLU>(),
                orientation = Quaternion.Euler(90, targetRotQ.eulerAngles.y, 0).To<FLU>()
            },
            // slight lift
            new RosMessageTypes.Geometry.Pose{
                position = (targetPos + liftOffset).To<FLU>(),
                orientation = Quaternion.Euler(90, targetRotQ.eulerAngles.y, 0).To<FLU>()
            },
            // turn motion will be inserted here
            new RosMessageTypes.Geometry.Pose{
                position = (targetPos + liftOffset).To<FLU>(),
                orientation = Quaternion.Euler(90, targetRotQ.eulerAngles.y, 0).To<FLU>()
            },
            // return to clench pose
            new RosMessageTypes.Geometry.Pose{
                position = (targetPos + pickLowerPoseOffset).To<FLU>(),
                orientation = Quaternion.Euler(90, targetRotQ.eulerAngles.y, 0).To<FLU>()
            },
            // retreat pose
            new RosMessageTypes.Geometry.Pose{ 
                position = approach.To<FLU>(),
                orientation = Quaternion.Euler(90, targetRotQ.eulerAngles.y, 0).To<FLU>()
            }};
        
        freeParams = new float[] {turnAngle};
            
        // Send the request and process the response
        yield return HandleRequestAndResponse(request,null);
        
        // Report success or failure via the callback function
        if (finishedCallback != null) finishedCallback(motionExecutionSuccess);
    }
    
    public IEnumerator HandleRequestAndResponse(MoverServiceRequest request, Action<GameObject> contactCallback){
        
        rosResponse = null;
        
        Debug.Log("sending request to ROS side...");
        
        // send trajectory planning request to ROS side
        ros.SendServiceMessage<MoverServiceResponse>(rosServiceName, request, TrajectoryResponse);
        
        // wait for response from ROS side
        while (rosResponse is null) yield return null;
        
        // check response content
        if (rosResponse.trajectories != null && rosResponse.trajectories.Length > 0){
            Debug.Log("Trajectory returned.");
            // start execution of planned trajectory
            yield return ExecuteTrajectories(rosResponse, contactCallback);
        }
        else{
            Debug.Log("No trajectory returned from MoverService.");
            motionExecutionSuccess = false;
        }
    }

    RosMessageTypes.Moveit.RobotTrajectory[] ReviseTurnDirection(RosMessageTypes.Moveit.RobotTrajectory[] trajectories){
        
        RosMessageTypes.Moveit.RobotTrajectory trajectory = trajectories[3];
        
        int length = trajectory.joint_trajectory.points.Length;
        var jointPosition = 0.0;
        float turnAngle = freeParams[0];
        float angleStep = (float)(Mathf.Deg2Rad*turnAngle)/(length-1);
        for (int i=1;i<length;i++){
            jointPosition = trajectory.joint_trajectory.points[i].positions[5];
            Debug.Log(i+" jointPosition: "+jointPosition);
            jointPosition = trajectory.joint_trajectory.points[i-1].positions[5]-angleStep;
            trajectory.joint_trajectory.points[i].positions[5] = jointPosition;
        }
        trajectories[3] = trajectory;
        
        for (int iPose = 4; iPose<6; iPose++){
            trajectory = trajectories[iPose];
            length = trajectory.joint_trajectory.points.Length;
            for (int i=0;i<length;i++){
                trajectory.joint_trajectory.points[i].positions[5] = jointPosition;
            }
            trajectories[iPose] = trajectory;
        }
        
        return trajectories;
    }
    
    
    // handler for the mover service response
    void TrajectoryResponse(MoverServiceResponse response){
        rosResponse = response;
    }

    /// <summary>
    ///     Execute the returned trajectories from the MoverService.
    ///
    ///     The expectation is that the MoverService will return a sequence of trajectory plans,
    ///     where each plan is an array of robot poses. A robot pose is the joint angle values
    ///     of the six robot joints.
    ///
    ///     Executing a single trajectory will iterate through every robot pose in the array while updating the
    ///     joint values on the robot.
    /// 
    /// </summary>
    /// <param name="response"> MoverServiceResponse received from ur3_moveit mover service running in ROS</param>
    /// <returns></returns>
    private IEnumerator ExecuteTrajectories(MoverServiceResponse response, Action<GameObject> contactCallback){
        
        executionAttempted = true;
        motionExecutionSuccess = true;
        
        if (response.trajectories != null){
            
            // For every trajectory plan returned
            Debug.Log("Executing trajectories. Number of trajectories:"+response.trajectories.Length);
            
            if (currentAffType == "turn"){
                response.trajectories = ReviseTurnDirection(response.trajectories);
                }
            
            for (int poseIndex  = 0 ; poseIndex < response.trajectories.Length; poseIndex++){
                
                int nSteps =1;
                switch(poseIndex){
                    case 0:
                        nSteps = 2;
                        break;
                    case 1:
                        nSteps = 10;
                        break;
                    case 2:
                        if (currentAffType == "turn") nSteps = 5;
                        else nSteps = 2;
                        break;
                    case 3:
                        nSteps = 100;
                        break;
                    case 4:
                        nSteps = 5;
                        break;
                    case 5:
                        nSteps = 2;
                        break;
                }
                
                // For every robot pose in trajectory plan
                int len = response.trajectories[poseIndex].joint_trajectory.points.Length;
                for (int jointConfigIndex  = 1 ; jointConfigIndex < response.trajectories[poseIndex].joint_trajectory.points.Length; jointConfigIndex++)
                {
                    var jointPositions = response.trajectories[poseIndex].joint_trajectory.points[jointConfigIndex].positions;
                    float[] result = jointPositions.Select(r=> (float)r * Mathf.Rad2Deg).ToArray();
                    
                    float[] prevTargets = new float[jointArticulationBodies.Length];
                    for (int joint = 0; joint < jointArticulationBodies.Length; joint++){
                        prevTargets[joint] = jointArticulationBodies[joint].xDrive.target;
                    }
                    
                    for (int iStep=1; iStep<=nSteps;iStep++){
                        float ratio = (float)iStep/nSteps;
                        for (int joint = 0; joint < jointArticulationBodies.Length; joint++)
                        {
                            var joint1XDrive  = jointArticulationBodies[joint].xDrive;
                            joint1XDrive.target = (1-ratio)*prevTargets[joint]+ratio*result[joint];
                            jointArticulationBodies[joint].xDrive = joint1XDrive;
                        }
                        
                        // Wait for robot to achieve pose for all joint assignments
                        yield return new WaitForSeconds(jointAssignmentWait);
                    }
                }
                
                // Wait for the robot to achieve the final pose from joint assignment
                yield return new WaitForSeconds(poseAssignmentWait);
                
                if (poseIndex == 1){
                    yield return IterateToGrip(!gripperHoldingObject, contactCallback);
                    if (currentAffType == "grasp"){
                        DetachStickies();
                    }
                }
                
                if (poseIndex == 4 && currentAffType == "turn")
                    yield return IterateToGrip(false, contactCallback);
            }
            
            yield return StartCoroutine(InitializeRobotPoseRoutine(false));
            yield return new WaitForSeconds(poseAssignmentWait);
            
            if (currentAffType == "grasp")
                RestoreStickies();
            
            if (currentAffType == "grasp" && motionExecutionSuccess)
                gripperHoldingObject = true;
            if (currentAffType == "place" && motionExecutionSuccess)
                gripperHoldingObject = false;
        }
    }

    /// <summary>
    ///     Find all robot joints in Awake() and add them to the jointArticulationBodies array.
    ///     Find all gripper joints and assign them to their respective articulation body objects.
    /// </summary>
    void Awake(){
        jointArticulationBodies = new ArticulationBody[numRobotJoints];
        string shoulder_link = "world/base_link/shoulder_link";
        jointArticulationBodies[0] = robot.transform.Find(shoulder_link).GetComponent<ArticulationBody>();

        string arm_link = shoulder_link + "/upper_arm_link";
        jointArticulationBodies[1] = robot.transform.Find(arm_link).GetComponent<ArticulationBody>();
        
        string elbow_link = arm_link + "/forearm_link";
        jointArticulationBodies[2] = robot.transform.Find(elbow_link).GetComponent<ArticulationBody>();
        
        string forearm_link = elbow_link + "/wrist_1_link";
        jointArticulationBodies[3] = robot.transform.Find(forearm_link).GetComponent<ArticulationBody>();
        
        string wrist_link = forearm_link + "/wrist_2_link";
        jointArticulationBodies[4] = robot.transform.Find(wrist_link).GetComponent<ArticulationBody>();
        
        string hand_link = wrist_link + "/wrist_3_link";
        jointArticulationBodies[5] = robot.transform.Find(hand_link).GetComponent<ArticulationBody>();

        articulationChain = robot.GetComponent<RosSharp.Control.Controller>().GetComponentsInChildren<ArticulationBody>();

        var gripperJointNames = new string[] { "right_outer_knuckle", "right_inner_finger", "right_inner_knuckle", "left_outer_knuckle", "left_inner_finger", "left_inner_knuckle" };
        gripperJoints = new List<ArticulationBody>();

        foreach (ArticulationBody articulationBody in robot.GetComponentsInChildren<ArticulationBody>()){
            if (gripperJointNames.Contains(articulationBody.name)){
                gripperJoints.Add(articulationBody);
            }
        }
    }

    void Start(){
        
        // Get ROS connection static instance
        ros = ROSConnection.instance;
        
        affordanceCheck = GameObject.Find("AffordanceChecker").GetComponent<AffordanceCheck>();
        
        // Find finger pads and parts
        ghostFingerPads = new List<GameObject>();
        ghostFingerPads.Add(GameObject.Find("LeftFingerFollow"));
        ghostFingerPads.Add(GameObject.Find("RightFingerFollow"));
        fingerPads = new List<GameObject>();
        fingerPads.Add(GameObject.Find("left_inner_finger_pad/Collisions/unnamed/LeftFingerCollisionBox"));
        fingerPads.Add(GameObject.Find("right_inner_finger_pad/Collisions/unnamed/RightFingerCollisionBox"));
        
        // Find the gripper base
        gripperBase = GameObject.Find("robotiq_arg2f_base_link_0_col");
        gripperFollowTarget = GameObject.Find("GripperFollowTarget");
        
        stickies = (StickyScript[]) GameObject.FindObjectsOfType(typeof(StickyScript));
    }
    
}
