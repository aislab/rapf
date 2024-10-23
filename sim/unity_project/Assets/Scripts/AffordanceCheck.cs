/*
Copyright 2024 Autonomous Intelligence and Systems (AIS) Lab, Shinshu University & EPSON AVASYS Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software 
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Vector3 = UnityEngine.Vector3;
using System.IO;
using System.Threading.Tasks;
using System;
using UnityEngine.SceneManagement;
using Unity.Robotics.ROSTCPConnector;
using System.Linq;


public class AffordanceCheck : MonoBehaviour{
    public int dataIndex = 0;
    public int dataStep = 0;
    public int sequenceLength;
    public bool enableSceneRandomisation;
    public bool bypassRobotMotion = true;
    public string dataSaveDir = "/DATA/SAVE/DIR/";
    public string planningTaskDir;
    public float XZ_DivisionFactorForPlanExecution = 1.075f;
    public float Y_LeewayForPlanExecution = 0.005f;
    public bool detectPlaceUsingMarkers;
    public bool detectPlaceUniform;
    public bool useTopDownCamForPlanning;
    public int n_frames_per_check = 1;
    public int maxAffExecutionAttempts = 8;
    public bool enableTurnAffordance = true;
    public float turnAffordancePreference = 0.0f;
    public bool discretisedSpread = true;
    public float spreadRangeX = 0.35f;
    private GameObject graspChecker;
    private Collider placeCheckerBlockCollider;
    private CollisionMemory graspCheckerMemory;
    private CollisionMemory placeCheckerBlockMemory;
    private CollisionMemory placeCheckerWristMemory;
    private GameObject roughPlaceChecker;
    private Vector3 roughPlaceCheckerRestPosition;
    private float placementGroundPlaneY;
    private GameObject[] tagged;
    private List<string> affTypes = new List<string>();
    private List<Vector3> affPositions = new List<Vector3>();
    private List<Vector3> affOrientations = new List<Vector3>();
    private List<GameObject> affTargetObject = new List<GameObject>();
    private List<bool> affSymmetry = new List<bool>();
    public Camera captureCamTop;
    public Camera captureCamFront;
    public CaptureCameraScript captureCamTopScript;
    public CaptureCameraScript captureCamFrontScript;
    private System.Random randomiser = new System.Random();
    private System.Random placement_randomiser;
    private TrajectoryPlanner trajectoryPlanner;
    public InputField affordanceNumberField;
    
    public float stabilityThreshold = 0.0000001f;
    public float collisionCheckWaitTime = 0.025f;
    public int placeCheckRes = 10;
    public bool fixedZ = true;
    
    public int rotationStep = 90; // should divide 360 without remainder
    
    public float defaultGraspObjectOverlap;
    
    public GameObject robotBase;
    public GameObject robot;
    public float minRobotReachability;
    public float maxRobotReachability;
    private Rect reachableRect;
    private Rect tooCloseRect;
    
    private bool holdingObject = false;
    
    // Handle on the finger pads
    private List<GameObject> fingerPads;
    
    private bool scenarioIterationComplete;
    [HideInInspector] public bool affordanceExecutionEnded;
    [HideInInspector] public bool affordanceExecutionSuccess;
    
    private ROSConnection ros;
    
    private GameObject placeChecker;
    private GameObject placeCheckerBlock;
    private GameObject placeCheckerWrist;
    private GameObject ghostObject;
    private Vector3 ghostObjectStandByPosition;
    private Quaternion ghostObjectStandByRotation;
    private Vector3 targetRotQ;
    
    private StabilityCheckTag[] stabilityCheckTags;
    
    float placeCheckerOffsetToMinY;
    Vector3 occlusionAreaCentre;
    float occlusionAreaSize;
    Vector2 visibilityCentre;
    
    private StickyScript[] stickies;
    
    private List<PositionRandomiserTag> tags;
    private List<GameObject> placedObjects;
    private List<PlacementConstraint> collisionConstraints;
    private Bounds planeBounds;
    private float stackingMargin = 0.001f;
    private SurfaceObjectPlacer placer;
    public GameObject plane;
    private ReachabilityConstraint maxReach;
    private ReachabilityConstraint minReach;
    public int maxPlacementTries = 100;
    public int n_support_objects_choices_per_object = 10;
    public int n_attempts_per_support_object = 5;
    
    private ReachabilityConstraint visibility;
    private bool heldObjectSymmetryState;
    private List<(float,float)> occupiedList;
    
    
    // Start is called before the first frame update
    void Start(){
        // Get ROS connection static instance
        ros = ROSConnection.instance;
        
        graspChecker = GameObject.Find("GraspCheckerPivot");
        graspCheckerMemory = GameObject.Find("GraspCheckerWrist").GetComponent<CollisionMemory>();
        placeCheckerBlockCollider = GameObject.Find("PlaceCheckerBlock").GetComponent<Collider>();
        placeCheckerBlockMemory = GameObject.Find("PlaceCheckerBlock").GetComponent<CollisionMemory>();
        placeCheckerWristMemory = GameObject.Find("PlaceCheckerWrist").GetComponent<CollisionMemory>();
        roughPlaceChecker = GameObject.Find("RoughPlaceChecker");
        Vector3 p = roughPlaceChecker.transform.position;
        roughPlaceCheckerRestPosition = new Vector3(p.x,p.y,p.z);
        
        placementGroundPlaneY = 0.65f;
        captureCamTop = GameObject.Find("CaptureCameraTop").GetComponent<Camera>();
        captureCamFront = GameObject.Find("CaptureCameraFront").GetComponent<Camera>();
        captureCamTopScript = captureCamTop.GetComponent<CaptureCameraScript>();
        captureCamFrontScript = captureCamFront.GetComponent<CaptureCameraScript>();
        trajectoryPlanner = GameObject.Find("Publisher").GetComponent<TrajectoryPlanner>();
        
        // Find fingers
        fingerPads = new List<GameObject>();
        fingerPads.Add(GameObject.Find("LeftFingerFollow"));
        fingerPads.Add(GameObject.Find("RightFingerFollow"));
        
        robotBase = GameObject.Find("base");
        
        placeChecker = GameObject.Find("PlaceCheckerPivot");
        placeCheckerBlock = GameObject.Find("PlaceCheckerBlock");
        placeCheckerWrist = GameObject.Find("PlaceCheckerWrist");
        
        ReachabilityConstraint maxReachConstraint = CreateReachabilityConstraint(robotBase.transform.position, maxRobotReachability, ReachabilityConstraint.LimitType.max);
        float x = maxReachConstraint.robotX - maxReachConstraint.robotReachabilityLimit;
        float z = maxReachConstraint.robotZ - maxReachConstraint.robotReachabilityLimit;
        float size = maxReachConstraint.robotReachabilityLimit * 2;
        reachableRect = new Rect(x, z, size, size);
        
        ReachabilityConstraint minReachConstraint = CreateReachabilityConstraint(robotBase.transform.position, minRobotReachability, ReachabilityConstraint.LimitType.min);
        x = minReachConstraint.robotX - minReachConstraint.robotReachabilityLimit;
        z = minReachConstraint.robotZ - minReachConstraint.robotReachabilityLimit;
        size = minReachConstraint.robotReachabilityLimit * 2;
        tooCloseRect = new Rect(x, z, size, size);
        
        stabilityCheckTags = FindObjectsOfType<StabilityCheckTag>();
        
        GameObject occlusionArea = GameObject.Find("OcclusionArea");
        occlusionAreaCentre = occlusionArea.transform.position;
        occlusionAreaSize = occlusionArea.transform.localScale.x/2;
        visibilityCentre = new Vector2(occlusionAreaCentre.x,occlusionAreaCentre.z);
        
        stickies = (StickyScript[]) GameObject.FindObjectsOfType(typeof(StickyScript));
        
        placer = new SurfaceObjectPlacer(plane, minReach, maxReach, visibility, maxPlacementTries);
        
        tags = ((PositionRandomiserTag[]) GameObject.FindObjectsOfType(typeof(PositionRandomiserTag))).ToList();
        
        StartCoroutine(trajectoryPlanner.InitializeRobotPoseRoutine(false));
    }
    
    
    public static ReachabilityConstraint CreateReachabilityConstraint(Vector3 robotBasePosition, float limit, ReachabilityConstraint.LimitType limitType){
        ReachabilityConstraint constraint = new ReachabilityConstraint();
        constraint.robotX = robotBasePosition.x;
        constraint.robotZ = robotBasePosition.z;
        constraint.limitType = limitType;
        constraint.robotReachabilityLimit = limit;
        return constraint;
    }
    
    
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
        Debug.Log("vertices:"+vertices.Length);
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
        Debug.Log("height:"+(maxY-minY));
        return new float[] {maxY-minY, minY, maxY};
    }
    
    
    private float FindMinY(GameObject obj){
        Mesh mesh = null;
        MeshFilter mF = obj.GetComponent<MeshFilter>();
        if (mF != null){
            mesh = mF.mesh;
        }
        Vector3[] vertices = mesh.vertices;
        Debug.Log("vertices:"+vertices.Length);
        float minY = float.MaxValue;
        for (int i = 1; i < vertices.Length; i++){
            Vector3 V = obj.transform.TransformPoint(vertices[i]);
            if (V[1] < minY){
                minY = V[1];
            }
        }
        Debug.Log("FindMinY: object y:"+obj.transform.position.y+" minY:"+minY+" rotation:"+obj.transform.rotation);
        return minY;
    }
    
    
    private string ConstructLine(string affType, Vector3 position, Vector3 rotation, bool symmetry, float[] freeParams=null){
        Vector3 pixelPositionTop = captureCamTop.WorldToScreenPoint(position);
        Vector3 pixelPositionFront = captureCamTop.WorldToScreenPoint(position);
        string str = affType+" position:"+position.ToString("R")+
                             " rotation:"+rotation.ToString("R")+
                             " symmetry:"+symmetry+
                             " screen_position_top:"+pixelPositionTop.ToString("R")+
                             " screen_position_front:"+pixelPositionFront.ToString("R");
        if (freeParams != null)
            str = str+" free_params:("+string.Join(", ",freeParams)+")";
        return str;
    }
    
    
    private IEnumerator SinglePlacabilityCheck(float x, float y, float z){
        
        CollisionMemory placeCheckerMemory = ghostObject.GetComponent<CollisionMemory>();
        CollisionMemory roughPlaceCheckerMemory = roughPlaceChecker.GetComponent<CollisionMemory>();
        float roughPlaceCheckerExtentY = roughPlaceChecker.GetComponent<Renderer>().bounds.extents.y;
        
        Vector2 xy = new Vector2(x,z);
        if (tooCloseRect.Contains(xy)) yield break;
        if (Math.Sqrt(x*x+z*z) > maxRobotReachability) yield break;
        if (Vector2.Distance(visibilityCentre, xy) < occlusionAreaSize) yield break;
        
        Vector3 xyz;
        List<Collider> colliderList;
        
        // rotation
        for (int r=0; r<360; r+=rotationStep){
            
            xyz = new Vector3(x,y+placeCheckerOffsetToMinY+0.0001f,z);
            ghostObject.transform.position = xyz;
            yield return new WaitForSeconds(collisionCheckWaitTime);
            colliderList = new List<Collider>();
            colliderList.AddRange(placeCheckerMemory.colliderList);
            colliderList.AddRange(placeCheckerBlockMemory.colliderList);
            colliderList.AddRange(placeCheckerWristMemory.colliderList);
            if (colliderList.Count > 0){
                placeCheckerMemory.ClearCollisionMemory();
                placeCheckerBlockMemory.ClearCollisionMemory();
                placeCheckerWristMemory.ClearCollisionMemory();
            }
            else{
                xyz = new Vector3(x,y,z);
                Vector3 rot = ghostObject.transform.rotation.eulerAngles;
                Debug.Log("placeable: xyz="+x+","+y+","+z+" r="+rot);
                affTypes.Add("place");
                affPositions.Add(xyz);
                affOrientations.Add(new Vector3(0,r,0));
                affSymmetry.Add(heldObjectSymmetryState);
            }
            ghostObject.transform.Rotate(0, rotationStep, 0, Space.World);
        }
        ghostObject.transform.position = ghostObjectStandByPosition;
        ghostObject.transform.rotation = ghostObjectStandByRotation;
    }
    
    
    private void ClearAffData(){
        // clear affordance data lists
        affTypes.Clear();
        affPositions.Clear();
        affOrientations.Clear();
        affTargetObject.Clear();
        affSymmetry.Clear();
    }
    
    
    private IEnumerator PlaceabilityCheckRoutine(){
        Debug.Log("Starting placeability check routine");
        
        // escape hatch for when ghostObject is missing (can happen with physically failed grasp?)
        if (ghostObject == null) yield break;

        float[] d = FindHeightAndRange(ghostObject);
        placeCheckerOffsetToMinY = ghostObject.transform.position.y-d[1];
        
        if (detectPlaceUsingMarkers){
            GameObject[] placeMarkers = GameObject.FindGameObjectsWithTag("PlaceabilityCheckTag");
            foreach (GameObject marker in placeMarkers){
                Vector3 pos = marker.transform.position;
                Vector3 rot = marker.transform.eulerAngles;
                Debug.Log("original marker rotation: "+rot);
                while (Math.Abs(rot.x) > 180) rot.x -= 360f*Math.Sign(rot.x);
                while (Math.Abs(rot.z) > 180) rot.z -= 360f*Math.Sign(rot.z);
                Debug.Log("normalised marker rotation: "+rot);
                if (Math.Abs(rot.x) < 5 && Math.Abs(rot.z) < 5){
                    Debug.Log("placeability check for marker: "+marker.name);
                    yield return SinglePlacabilityCheck(pos.x,pos.y+0.001f,pos.z);
                }
                Debug.Log("placeability check skips marker: "+marker.name+" (excessive tilt: "+rot+")");
            }
        }
        if (detectPlaceUniform){
            for (int ix=0; ix<=placeCheckRes; ix++){
                float x = reachableRect.x+ix*reachableRect.width/placeCheckRes;
                
                for (int iz=0; iz<=(fixedZ?0:placeCheckRes); iz++){
                    float z = fixedZ?-0.25f:(reachableRect.y+iz*reachableRect.height/placeCheckRes);
                    
                    yield return SinglePlacabilityCheck(x,placementGroundPlaneY,z);
                }
            }
        }
        Debug.Log("Placeability check routine completed");
    }
    
    
    private IEnumerator GripperAccessCheckRoutine(string affType){
        Debug.Log("GripperAccessCheckRoutine for aff type: "+affType);
        
        yield return CheckSceneStable();
        
        // initialise new check
        graspCheckerMemory.ClearCollisionMemory();
        
        if (affType == "grasp")
            tagged = GameObject.FindGameObjectsWithTag("GraspabilityCheckTag");
        if (affType == "turn")
            tagged = GameObject.FindGameObjectsWithTag("TurnabilityCheckTag");
            
        Debug.Log("Found "+tagged.Length+" active objects.");
        
        float angle = 0f;
        bool useCustomAngles = false;
        int customAngleIndex = 0;
        float[] customGraspAngles = new float[0];
        
        for (int current_object_index=0;current_object_index<tagged.Length;current_object_index++){
            
            if (!tagged[current_object_index].activeSelf) continue;
            
            Vector3 pos = tagged[current_object_index].transform.position;
            Vector3 checkerPosition;
            
            GameObject targetObject = null;
            if (affType == "grasp")
                targetObject = tagged[current_object_index];
            if (affType == "turn")
                targetObject = tagged[current_object_index].transform.parent.gameObject;
            
            float x = pos.x;
            float z = pos.z;
            Vector2 xy = new Vector2(x,z);
            if (tooCloseRect.Contains(xy)) continue;
            if (Math.Sqrt(x*x+z*z) > maxRobotReachability) continue;
            if (Vector2.Distance(visibilityCentre, xy) < occlusionAreaSize) continue;
            
            float[] d = FindHeightAndRange(targetObject);
            float h = d[0];
            float maxY = d[2];
            float graspY = maxY-Math.Min(h,defaultGraspObjectOverlap);
            checkerPosition = new Vector3(pos.x,graspY,pos.z);
            Debug.Log("Object "+current_object_index+" within reach --> running graspability check");

            bool symmetry;
            GraspAngleScript graspAngleScript = tagged[current_object_index].GetComponent<GraspAngleScript>();
            if (graspAngleScript != null){
                useCustomAngles = true;
                (customGraspAngles,symmetry) = graspAngleScript.GetGraspAngles();
                Debug.Log(tagged[current_object_index].name+" suggests grasp angles");
                foreach (float a in customGraspAngles)Debug.Log(a);
                angle = customGraspAngles[0];
                customAngleIndex = 0;
            }
            else{
                symmetry = true;
                useCustomAngles = false;
                if (tagged[current_object_index].GetComponent<Collider>() is SphereCollider){
                    angle = 0f;
                }
                else{
                    Debug.LogError("Objects with a collider other than SphereCollider must have a GraspAngleScript attached.");
                    Debug.Break();
                }
            }
            
            graspChecker.transform.position = checkerPosition;
            
            while (true){
            
                graspChecker.transform.rotation = Quaternion.Euler(0, angle, 0);
                
                graspCheckerMemory.ClearCollisionMemory();
                yield return new WaitForSeconds(collisionCheckWaitTime);
                
                if (!graspCheckerMemory.has_collided){
                    Vector3 xyz = graspChecker.transform.position;
                    Vector3 rot = graspChecker.transform.rotation.eulerAngles;
                    if (xyz.y > 0){
                        Debug.Log("graspable: object:"+current_object_index+" position:"+xyz.ToString("R")+" angle:"+angle);
                        affTypes.Add(affType);
                        affPositions.Add(xyz);
                        affOrientations.Add(new Vector3(0,angle,0));
                        affTargetObject.Add(targetObject);
                        affSymmetry.Add(symmetry);
                    }
                }

                if (useCustomAngles){
                    customAngleIndex += 1;
                    if (customAngleIndex==customGraspAngles.Length) break;
                    else{
                        angle = customGraspAngles[customAngleIndex];
                        graspChecker.transform.rotation = Quaternion.Euler(0,angle,0);
                    }
                }
                else{
                    angle += rotationStep;
                    if (angle == 360) break;
                    graspChecker.transform.rotation = Quaternion.Euler(0,angle,0);
                }
            }
        }
        graspCheckerMemory.ClearCollisionMemory();
        graspChecker.transform.position = new Vector3(-0.5f, 0.0f, 0f);
    }
    
    
    private void WriteAffData(){
        Debug.Log("Writing affordances to dataset");
        List<string> lines = new List<string>();
        for (int i=0; i<affTypes.Count; i++){
            lines.Add(ConstructLine(affTypes[i],affPositions[i],affOrientations[i],affSymmetry[i]));
        }
        File.WriteAllLines(dataSaveDir+"/"+dataIndex+"_"+dataStep+"_affordances.txt", lines);
        File.WriteAllLines(dataSaveDir+"/_affordances.txt", lines);
    }
    
    
    private IEnumerator WriteStateData(){
        Debug.Log("Writing RGBD capture to dataset");
        yield return captureCamTopScript.CaptureRoutine(dataSaveDir+"/"+dataIndex+"_"+dataStep+"_");
        yield return captureCamFrontScript.CaptureRoutine(dataSaveDir+"/"+dataIndex+"_"+dataStep+"_");
    }
    
    
    private IEnumerator CheckSceneStable(){
        Debug.Log("Starting stability check");
        int iteration = 0;
        List<Vector3> positions = new List<Vector3>();
        foreach (StabilityCheckTag tag in stabilityCheckTags){
            Vector3 p = tag.gameObject.transform.position;
            positions.Add(new Vector3(p.x,p.y,p.z));
        }
        yield return new WaitForSeconds(0.1f);
        
        int nStableFrames=0;
        while (true){
            
            bool stable = true;
            for (int i = 0; i < stabilityCheckTags.Length; i++){
                Vector3 p = stabilityCheckTags[i].transform.position;
                if (p.y<1.3){ // objects above this height are currently being held and may jitter, so we ignore them    
                    float dist = Vector3.Distance(p,positions[i]);
                    positions[i] = new Vector3(p.x,p.y,p.z);
                    if (p.y > 0){ // ignore objects that fell off the world
                        stable = stable && (dist < stabilityThreshold);
                    }
                }
            }
            if (stable){
                Debug.Log("Scene stable");
                nStableFrames+=1;
                if (nStableFrames==5) break;
            }
            else{
                nStableFrames=0;
            }
            yield return new WaitForSeconds(0.1f);
            iteration++;
        }
    }
    
    
    public void MotionFinishedCallback(bool success){
        affordanceExecutionSuccess = success;
    }
    
    
    public void ContactCallback(GameObject obj){
        if (ghostObject != null){
            placeChecker.transform.SetParent(null);
            Destroy(ghostObject);
        }
        
        Debug.Log("Instantiating ghost object for:"+obj.name);
        ghostObject = GameObject.Instantiate(obj);
        ghostObject.tag = "Untagged";
        
        Collider ghostObjectCollider;
        ghostObjectCollider = ghostObject.GetComponent<MeshCollider>();
        if (ghostObjectCollider==null){
            ghostObjectCollider = ghostObject.GetComponent<Collider>();
        }
        else{
            BoxCollider bc = ghostObject.GetComponent<BoxCollider>();
            if (bc!=null)Destroy(bc);
            SphereCollider sc = ghostObject.GetComponent<SphereCollider>();
            if (sc!=null)Destroy(sc);
        }

        ghostObject.GetComponent<Rigidbody>().isKinematic = true;
        Vector3 p = placeChecker.transform.position;
        
        float ghostMinY = FindMinY(obj);
        float ghostExtentY = ghostObject.transform.position.y - ghostMinY;
        
        ghostObject.transform.position = new Vector3(p.x,p.y+ghostExtentY-trajectoryPlanner.placementObjectOffsetY,p.z);
        ghostObject.transform.Rotate(0f,-targetRotQ.y,0f,Space.World);
        ghostObjectStandByPosition = ghostObject.transform.position;
        ghostObjectStandByRotation = ghostObject.transform.rotation;
        placeChecker.transform.SetParent(ghostObject.transform);
        placeChecker = GameObject.Find("PlaceCheckerPivot");
        ghostObjectCollider.isTrigger = true;
        ghostObjectCollider.GetComponent<Rigidbody>().useGravity = false;
        Physics.IgnoreCollision(placeCheckerWrist.GetComponent<Collider>(),ghostObjectCollider,true);
        Physics.IgnoreCollision(placeCheckerBlock.GetComponent<Collider>(),ghostObjectCollider,true);
        ghostObject.AddComponent<CollisionMemory>();
    }
    
    
    public IEnumerator RunAffordanceRoutine(int affIndex=-1, float[] freeParams=null, bool writeData=false, int retries=1){
        
        if (affPositions.Count == 0){
            Debug.Log("Affordance list is empty...");
            yield break;
        }
        
        Debug.Log("Executing affordance");
        
        affordanceExecutionEnded = false;
        Debug.Log("Affordance index: "+affIndex+"/"+affPositions.Count);
        
        string affType = affTypes[affIndex];
        Debug.Log("Affordance parameters: type="+affTypes[affIndex]+" pos="+affPositions[affIndex]);
        
        for (int i=0; i<retries; i++){
            if (affType == "place"){
                yield return trajectoryPlanner.PlaceHeldObject(affPositions[affIndex],affOrientations[affIndex],MotionFinishedCallback);
                heldObjectSymmetryState = false;
            }
            if (affType == "grasp"){
                if (ghostObject!=null){
                    Debug.Log("Destroying existing ghost object: "+ghostObject.name);
                    placeChecker.transform.SetParent(null);
                    Destroy(ghostObject);
                }
                targetRotQ = affOrientations[affIndex]; // used to positioning ghostobject elsewhere
                yield return trajectoryPlanner.PickAndHold(affPositions[affIndex],affOrientations[affIndex],affTargetObject[affIndex],MotionFinishedCallback,ContactCallback);
                heldObjectSymmetryState = affSymmetry[affIndex];
                }
            if (affType == "turn"){
                yield return trajectoryPlanner.TurnObject(affPositions[affIndex],affOrientations[affIndex],freeParams[0],affTargetObject[affIndex],MotionFinishedCallback);
                heldObjectSymmetryState = false;
            }
            if (affordanceExecutionSuccess) break;
        }
        
        if (affordanceExecutionSuccess && writeData){
            Debug.Log("Writing selected affordance to dataset");
            string str = ConstructLine(affTypes[affIndex],affPositions[affIndex],affOrientations[affIndex],affSymmetry[affIndex],freeParams);
            File.WriteAllText(dataSaveDir+"/"+dataIndex+"_"+dataStep+"_ExecutedAffordance.txt", str);
            Debug.Log("Received execution completed signal --> waiting for scene to stabilise");
            yield return CheckSceneStable();
            Debug.Log("Scene stable after affordance execution attempt");
            holdingObject = !holdingObject;
            dataStep++;
            Debug.Log("increased dataStep to "+dataStep);
        }
    }
    
    
    public void ScenarioIterationEnd(){
        scenarioIterationComplete = true;
    }
    
    
    public void FullAutoButtonPressed(){
        StartCoroutine(FullAutoRoutine());
    }
    
    
    public void Finished(bool success){
        Debug.Log("callback: finished / success: "+success);
        Debug.Break();
    }
    
    
    public void SendStateForPlanning(){
        Directory.CreateDirectory(planningTaskDir+"/start");
        if (useTopDownCamForPlanning)
            StartCoroutine(captureCamTopScript.CaptureRoutine(planningTaskDir+"/start/unity_state_"));
        else
            StartCoroutine(captureCamFrontScript.CaptureRoutine(planningTaskDir+"/start/unity_state_"));
    }
    
    
    public void SendStateAsGoal(){
        Directory.CreateDirectory(planningTaskDir+"/goal");
        if (useTopDownCamForPlanning)
            StartCoroutine(captureCamTopScript.CaptureRoutine(planningTaskDir+"/goal/unity_state_"));
        else
            StartCoroutine(captureCamFrontScript.CaptureRoutine(planningTaskDir+"/goal/unity_state_"));
    }
    
    
    public void ExecutePlannedAffordance(){
        if (bypassRobotMotion){
            Debug.LogWarning("[Execute Plan] button pressed with [Bypass Robot Motion] enabled. "+
                             "Plan execution is not compatible with motion bypass. Motion bypass has been disabled. "+
                             "Make sure the ROS side is launched for motion planning.");
            bypassRobotMotion = false;
        }
        StartCoroutine(ExecutePlannedAffordanceRoutine());
    }
    
    
    public IEnumerator ExecutePlannedAffordanceRoutine(){
        int step = 0;
        using (TextReader reader = File.OpenText(planningTaskDir+"/solution/plan.txt")){
            while (true){
                string line = reader.ReadLine();
                Debug.Log("read line: "+line);
                if (line == null) break;
                string[] vv = line.Split(' ');
                int affClass = int.Parse(vv[0]);
                float x = float.Parse(vv[1]);
                float z = -float.Parse(vv[2]);
                float y = float.Parse(vv[3]);
                float angle = float.Parse(vv[4]);
                float free = float.Parse(vv[5]);
                affTypes.Clear();
                affPositions.Clear();
                affOrientations.Clear();
                affTargetObject.Clear();
                affSymmetry.Clear();

                string affClassString = "UNKNOWN";
                switch (affClass){
                    case 0: 
                        affClassString = "grasp";
                        break;
                    case 1: 
                        affClassString = "place";
                        break;
                    case 2: 
                        affClassString = "turn";
                        break;
                }
                affTypes.Add(affClassString);
                affPositions.Add(new Vector3(x/XZ_DivisionFactorForPlanExecution,
                                             y+Y_LeewayForPlanExecution,
                                             z/XZ_DivisionFactorForPlanExecution));
                angle = (360*angle+180)%360-180; // normalise into [-180,180]
                affOrientations.Add(new Vector3(0,angle,0));
                affTargetObject.Add(null);
                free = -90*free;
                affSymmetry.Add(false); // symmetry state unknown so assume false
                Debug.Log("Executing planning result: type="+affClassString+"("+affClass+") x="+x+" y="+y+" z="+z+" angle:"+angle+" free:"+free);
                float[] freeParams = {free};
                yield return RunAffordanceRoutine(0,freeParams,false,30);
                if (!affordanceExecutionSuccess){
                    Debug.Log("Execution of planned affordance failed...");
                    yield break;
                }
                
                // capture result of each step
                StartCoroutine(captureCamTopScript.CaptureRoutine(planningTaskDir+"/solution/unity_result_"+step));
                yield return null;
                StartCoroutine(captureCamFrontScript.CaptureRoutine(planningTaskDir+"/solution/unity_result_"+step));
                step += 1;
            }
            Debug.Log("Plan execution completed");
        }
    }
    
    
    private void RestoreDrag(){
        PositionRandomiserTag[] tags = (PositionRandomiserTag[]) GameObject.FindObjectsOfType(typeof(PositionRandomiserTag));
        foreach (PositionRandomiserTag tag in tags){
            Rigidbody rb = tag.gameObject.GetComponent<Rigidbody>();
            rb.drag = tag.originalDrag;
        }
        Debug.Log("drag restored to original values");
    }
    
    
    private void ActivateStickies(){
        foreach (StickyScript sticky in stickies){
            sticky.Activate();
        }
    }
    
    
    public bool Tilted(Vector3 rot){
        while (Math.Abs(rot.x) > 180) rot.x -= 360f*Math.Sign(rot.x);
        while (Math.Abs(rot.z) > 180) rot.z -= 360f*Math.Sign(rot.z);
        Debug.Log("x,z rot: "+rot.x+", "+rot.z);
        return (Math.Abs(rot.x)>5 || Math.Abs(rot.z)>5);
    }
    
    
    public void RandomizePositions(){
        tags = tags.OrderBy(x => x.priority).ToList();
        placer.IterationStart();

        (List<GameObject> reachableObjects, List<GameObject> otherObjects) = SeparateTags(tags);

        Shuffle<GameObject>(reachableObjects);
        collisionConstraints = new List<PlacementConstraint>();
        placedObjects = new List<GameObject>();
        
        Dictionary<string,int> groupCounts = new Dictionary<string,int>();
        foreach (PositionRandomiserTag tag in tags){
            if (!tag.gameObject.activeSelf) continue;
            Debug.Log(tag.gameObject+" in randomiser group "+tag.randomisationGroup+" with priority "+tag.priority);
            if (tag.randomisationGroup!=""){
                if (groupCounts.ContainsKey(tag.randomisationGroup)){
                    groupCounts[tag.randomisationGroup] += 1;
                }
                else{
                    groupCounts[tag.randomisationGroup] = 1;
                }
            }
        }
        
        Dictionary<string,int> groupCountTargets = new Dictionary<string,int>();
        Dictionary<string,int> groupCountCurrent = new Dictionary<string,int>();
        foreach(KeyValuePair<string,int> kvp in groupCounts){
            groupCountTargets[kvp.Key] = placement_randomiser.Next(kvp.Value+1);
            groupCountCurrent[kvp.Key] = 0;
        }
        
        foreach (PositionRandomiserTag tag in tags){
            if (tag.randomisationGroup!=""){
                if (groupCountCurrent[tag.randomisationGroup] == groupCountTargets[tag.randomisationGroup]){
                    tag.gameObject.SetActive(false);
                    continue;
                }
                else{
                    groupCountCurrent[tag.randomisationGroup] += 1;
                }
            }
            else if (tag.presenceProbability < 1.0){
                if (placement_randomiser.NextDouble()>tag.presenceProbability){
                    tag.gameObject.SetActive(false);
                    continue;
                }
            }
            
            Rigidbody rb = tag.gameObject.GetComponent<Rigidbody>();
            tag.originalDrag = rb.drag;
            rb.drag = 15;
        }
        
        float yOffset = 0.1f;
        int n_spread = 0;
        float rx_sum = 0;
        List<float> spread_x_coords = new List<float>();
        foreach (PositionRandomiserTag tag in tags){
            if(!tag.gameObject.activeSelf)continue;
            if(tag.spread)n_spread += 1;
        }
        
        int n_slots = 0;
        bool[] occupied = null;
        if (discretisedSpread){
            n_slots = (int)Math.Floor(2*spreadRangeX/0.1f);
            occupied = new bool[n_slots];
        }
        else{
            for (int i=0;i<n_spread+1;i++){
                float rx = 0.1f+(float)placement_randomiser.NextDouble();
                rx_sum += rx;
                spread_x_coords.Add(rx_sum);
            }
        }
        int i_spread = 0;
        planeBounds = placer.plane.GetComponent<Renderer>().bounds;
        float y = planeBounds.center.y;
        foreach (PositionRandomiserTag tag in tags){
            if (!tag.gameObject.activeSelf) continue;
            if (tag.spread){
                GameObject obj = tag.gameObject;
                Bounds objBounds = obj.GetComponent<Renderer>().bounds;
                float heightAbovePlane = planeBounds.extents.y+objBounds.extents.y+stackingMargin;
                float x;
                if (discretisedSpread){
                    while (true){
                        int i_slot = placement_randomiser.Next(n_slots);
                        if (!occupied[i_slot]){
                            x = (i_slot+0.5f)-(n_slots/2.0f);
                            x *= 0.1f;
                            occupied[i_slot] = true;
                            break;
                        }
                    }
                }
                else{
                    x = -spreadRangeX+2*spreadRangeX*(spread_x_coords[i_spread]/rx_sum);
                }
                    
                obj.transform.position = new Vector3(x, y+heightAbovePlane, -0.3f);
                i_spread += 1;
                placedObjects.Add(obj);
            }
        }
        if (n_spread>0) yOffset += 0.1f;
        
        occupiedList = new List<(float,float)>();
        foreach (PositionRandomiserTag tag in tags){
            yOffset += PlaceObjectByTag(tag,yOffset);
        }
        
        foreach (PositionRandomiserTag tag in tags){
            if (!tag.gameObject.activeSelf) continue;
            GameObject obj = tag.gameObject;
            Rigidbody rb = tag.gameObject.GetComponent<Rigidbody>();
            rb.isKinematic = false;
        }
        
        ScenarioIterationEnd();
    }
    
    
    public float PlaceObjectByTag(PositionRandomiserTag tag, float yOffset){
        if (!tag.gameObject.activeSelf) return 0f;
        if (tag.spread) return 0f;
        GameObject obj = tag.gameObject;
        
        if (placement_randomiser.NextDouble()>=1.00)
        {
            obj.SetActive(false);
            return 0f;
        }
        
        obj.SetActive(true);
        Rigidbody rigidbody = obj.GetComponent<Rigidbody>();
        rigidbody.isKinematic = false;
        rigidbody.velocity = Vector3.zero;
        rigidbody.angularVelocity = Vector3.zero;
        
        bool success = false;
        if (tag.placeAroundInitialPosition){
            float d = (float)placement_randomiser.NextDouble()*tag.rangeAroundInitialPosition;
            double angle = placement_randomiser.NextDouble()*360.0;
            Bounds objBounds = obj.GetComponent<Renderer>().bounds;
            float heightAbovePlane = planeBounds.extents.y+objBounds.extents.y+stackingMargin;
            obj.transform.position = tag.initialPosition + new Vector3((float)Math.Sin(angle)*d,heightAbovePlane+yOffset,(float)Math.Cos(angle)*d);
            obj.transform.rotation = tag.initialRotation;
            placedObjects.Add(obj);
            success = true;
        }
        else if (tag.stackable && placedObjects.Count==1){
            // inefficiency: n_support_objects_choices_per_object is overkill when there are fewer placed objects than that
            for (int i=0; i<n_support_objects_choices_per_object; i++){
                int object_index = placement_randomiser.Next(placedObjects.Count);
                GameObject supportingObject = placedObjects[object_index];
                Debug.Log("Attempting placement of "+obj.name+" on top of "+supportingObject.name);
                
                Bounds objBounds = obj.GetComponent<Renderer>().bounds;
                float heightAbovePlane = planeBounds.extents.y+objBounds.extents.y+stackingMargin;
                Vector3 p = supportingObject.transform.position;
                obj.transform.position = new Vector3(p.x,p.y+0.1f,p.z);
                success = true;
                
                if (success){
                    placedObjects.Add(obj);
                    break;
                }
            }
        }
        
        if (!success && tag.placeInPlacementArea){
            Vector3 areaExtents = tag.placementAreaObject.GetComponent<Renderer>().bounds.extents;
            Vector3 areaPos = tag.placementAreaObject.transform.position;
            
            float x = 0;
            float z = 0;
            for (int i=0;i<100;i++){
                x = areaPos.x-areaExtents.x+2*areaExtents.x*(float)placement_randomiser.NextDouble();
                z = areaPos.z-areaExtents.z+2*areaExtents.z*(float)placement_randomiser.NextDouble();
                
                float dx = tag.discretisationStepX*(float)Math.Round(x/tag.discretisationStepX);
                if (tag.discretisedX){
                    if (tag.discretisationNoise) x = dx+0.05f*((float)placement_randomiser.NextDouble()-0.5f);
                    else x = dx;
                }
                
                float dz = tag.discretisationStepZ*(float)Math.Round(z/tag.discretisationStepZ);
                if (tag.discretisedZ){
                    if (tag.discretisationNoise) z = dz+0.05f*((float)placement_randomiser.NextDouble()-0.5f);
                    else z = dz;
                }
                
                if(!occupiedList.Contains((dx,dz))){
                    if(tag.doNotStackOnThisObject)occupiedList.Add((dx,dz));
                    break;
                }
            }
            
            obj.transform.position = new Vector3(x,areaPos.y+yOffset,z);
            obj.transform.rotation = tag.initialRotation;
            placedObjects.Add(obj);
            success = true;
        }
        
        if (!success){   
            Debug.Log("Attempting placement of "+obj.name+" on table surface");
            List<PlacementConstraint> newCollisionConstraints;
            (success, newCollisionConstraints) = placer.PlaceObject(obj, collisionConstraints, tag.mustBeReachable,yOffset);
            if (success){
                collisionConstraints.AddRange(newCollisionConstraints);
                placedObjects.Add(obj);
            }
        }
        
        if(tag.randomiseRotationY){
            if (360%tag.rotationStepY!=0){
                Debug.LogError("Bad rotationStepY setting: rotationStepY should cleanly divide 360");
                Debug.Break();
            }
            int n_angles = (int)(360/tag.rotationStepY);
            int i = placement_randomiser.Next(n_angles);
            Quaternion rot = tag.gameObject.transform.rotation;
            tag.gameObject.transform.rotation = Quaternion.Euler(rot.x,rot.y+i*tag.rotationStepY,rot.z);
        }
        
        float h = 0;
        if(!tag.doNotStackOnThisObject)
            h = FindHeight(obj)+0.01f;

        return h;
    }
    
    
    private (List<GameObject> reachableObjects, List<GameObject> otherObjects) SeparateTags(List<PositionRandomiserTag> tags){
        List<GameObject> reachableObjects = new List<GameObject>();
        List<GameObject> otherObjects = new List<GameObject>();

        foreach (PositionRandomiserTag tag in tags)
        {
            GameObject obj = tag.gameObject;
            if (tag.mustBeReachable){
                reachableObjects.Add(obj);
            }
            else{
                otherObjects.Add(obj);
            }
        }
        return (reachableObjects, otherObjects);
    }
    
    
    private static void Shuffle<T>(IList<T> ts){
        var count = ts.Count;
        var last = count - 1;
        for (var i = 0; i < last; ++i) {
            var r = UnityEngine.Random.Range(i, count);
            var tmp = ts[i];
            ts[i] = ts[r];
            ts[r] = tmp;
        }
    }
    
    
    public IEnumerator FullAutoRoutine(){
        DontDestroyOnLoad(ros);
        DontDestroyOnLoad(this.gameObject);
        
        Directory.CreateDirectory(dataSaveDir);
        
        while (true){
            
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
            yield return null;
            yield return null;
            yield return null;

            
            int repeats_per_state = 1;
            int iteration_seed = dataIndex/repeats_per_state;
            placement_randomiser = new System.Random(iteration_seed);
            Start();
            if (enableSceneRandomisation){
                RandomizePositions();
            }
            yield return trajectoryPlanner.InitializeRobotPoseRoutine(false);
            
            holdingObject = false;
            
            dataIndex++;
            dataStep = 0;
            Debug.Log("Data index incremented to: "+dataIndex);
            
            if (enableSceneRandomisation)
                while (!scenarioIterationComplete)
                    yield return null;
            
            yield return CheckSceneStable();
            GameObject colouredBlock = GameObject.Find("4x1x1");
            if (colouredBlock != null){
                Vector3 rot = colouredBlock.transform.eulerAngles;
                if (Tilted(rot)){
                    Debug.Log("Block tilted too much");
                    continue;
                }
            }
            
            while (true){
                Debug.Log("Start stability check");
                yield return CheckSceneStable();
                // check if cups are upright
                bool allUpright = true;
                foreach (PositionRandomiserTag tag in tags){
                    if (!tag.gameObject.activeSelf) continue;
                    if (tag.startUpright){
                        if (Tilted(tag.gameObject.transform.eulerAngles)){
                            Debug.Log(tag.gameObject+" not upright");
                            PlaceObjectByTag(tag,0.2f);
                            allUpright = false;
                        }
                    }
                }
                if (allUpright)break;
            }
            
            // check coloured block again in case cup re-randomisation tilted it
            if (colouredBlock != null){
                Vector3 rot = colouredBlock.transform.eulerAngles;
                if (Tilted(rot)){
                    Debug.Log("Block tilted too much");
                    continue;
                }
            }
            
            RestoreDrag();
            ActivateStickies();
            
            dataStep = 0;
            for (int i_step=0; i_step<sequenceLength; i_step++){
                
                ClearAffData();
                
                if (trajectoryPlanner.gripperHoldingObject){
                    Debug.Log("Start place affordance detection");
                    yield return PlaceabilityCheckRoutine();
                }
                else{ // gripper empty
                    Debug.Log("Start grasp affordance detection");
                    yield return GripperAccessCheckRoutine("grasp");
                    
                    if (enableTurnAffordance){
                        Debug.Log("Start turn affordance detection");
                        yield return GripperAccessCheckRoutine("turn");
                    }
                }
                if (affPositions.Count==0){
                    Debug.Log("No affordances found");
                    break;
                }
                Debug.Log("List of affordance types:");
                foreach (string t in affTypes){
                    Debug.Log(t);
                }
                
                if (affTypes.Count==0){
                    break;
                }
                
                WriteAffData();
                if (i_step == 0) yield return WriteStateData();
                
                if (trajectoryPlanner.gripperHoldingObject){
                    float minY = 1000.0f;
                    float maxY = -1000.0f;
                    foreach (Vector3 pos in affPositions){
                        if (pos.y < minY) minY=pos.y;
                        if (pos.y > maxY) maxY=pos.y;
                    }
                    for (int attempt=0; attempt<maxAffExecutionAttempts; attempt++){
                        
                        int index;
                        while (true){
                            index = randomiser.Next(0,affPositions.Count);
                            float keepProb = maxY>minY?(0.2f+0.8f*(affPositions[index].y-minY)/(maxY-minY)):1.0f;
                            if (randomiser.NextDouble()<keepProb){
                                Debug.Log("Accepted selected place affordance at y="+affPositions[index].y+" prob="+keepProb+" y range=["+minY+","+maxY+"]");
                                break;
                            }
                            Debug.Log("Rejected selected place affordance at y="+affPositions[index].y+" prob="+keepProb+" y range=["+minY+","+maxY+"]");
                        }
                        
                        Debug.Log("Start execution of place affordance "+index);
                        yield return RunAffordanceRoutine(index,null,true,2);
                        if (affordanceExecutionSuccess) break;
                        Debug.Log("Attempt "+attempt+": Execution of affordance "+index+" failed");
                    }
                }
                else{
                    for (int attempt=0; attempt<maxAffExecutionAttempts; attempt++){
                        int index = -1;
                        if (turnAffordancePreference>0){ // always picks the first turn affordance in the list (not suitable for scenes with multiple turn affordances)
                            for (int i=1; i<affPositions.Count; i++){
                                if (affTypes[i] == "turn" && randomiser.NextDouble()<turnAffordancePreference){
                                    index = i;
                                    break;
                                }
                            }
                        }
                        
                        if (index==-1){
                            int n_objects = 1;
                            List<int> objectFirstEntry = new List<int>();
                            objectFirstEntry.Add(0);
                            List<int> n_entries = new List<int>();
                            n_entries.Add(1);
                            for (int i=1; i<affPositions.Count; i++){
                                if (affTargetObject[i] != affTargetObject[i-1]){
                                    n_objects += 1;
                                    objectFirstEntry.Add(i);
                                    n_entries.Add(1);
                                }
                                else{
                                    n_entries[n_objects-1] += 1;
                                }
                            }
                            int objectIndex = randomiser.Next(0,n_objects);
                            int entryIndex = randomiser.Next(0,n_entries[objectIndex]);
                            index = objectFirstEntry[objectIndex]+entryIndex;
                            Debug.Log("Selected affordance: object: "+(objectIndex+1)+"/"+n_objects+" entry: "+(entryIndex+1)+"/"+n_entries[objectIndex]+" -> affordance: "+index+"/"+affPositions.Count);
                        }
                        
                        float turnAngle = 2*(randomiser.Next(0,2)-0.5f)*90;
                        float[] freeParams = (affTypes[index] == "turn") ? new float[] {turnAngle} : null; // will be ignored for grasp
                        Debug.Log("Start execution of grasp affordance "+index);
                        yield return RunAffordanceRoutine(index,freeParams,true,2);
                        
                        if (affordanceExecutionSuccess) break;
                        Debug.Log("Attempt "+attempt+": Execution of affordance "+index+" failed");
                        if (trajectoryPlanner.executionAttempted) break;
                    }
                }
                    
                if (affordanceExecutionSuccess){
                    yield return WriteStateData();
                }
                else{
                    Debug.Log("All execution attempts failed");
                    break;
                }
            }
        }
    }
    
    
    public void CaptureAffordanceOutcome(){
        Debug.Log("Saving RGBD capture");
        StartCoroutine(captureCamTopScript.CaptureRoutine(planningTaskDir+"/solution/execution_result_top"));
        StartCoroutine(captureCamFrontScript.CaptureRoutine(planningTaskDir+"/solution/execution_result_front"));
    }
}
