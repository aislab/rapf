/*
This file modifies the SurfaceObjectPlacer.cs file distributed by the Unity Robotics Hub.
*/


using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;


public class PlacementConstraint{
    /* Determines if a particular object placement on the x-z plane is valid. See
     subclasses for details. */

    public virtual bool Passes(float placementX, float placementZ, float objectRadius){
        return true;
    }
    
}

public class CollisionConstraint : PlacementConstraint{
    /* Checks if objects are placed far enough apart, so they cannot collide. */

    public float x;
    public float z;
    public float radius;

    public override bool Passes(float placementX, float placementZ, float objectRadius){
        Vector2 placementPt = new Vector2(placementX, placementZ);
        Vector2 basePt = new Vector2(x, z);
        float distance = Vector2.Distance(placementPt, basePt);
        float minDistance = objectRadius + radius;
        return distance > minDistance;
    }
    
}

public class ReachabilityConstraint : PlacementConstraint{
    /* Checks if an object is placed close enough to the robot arm to be reached. */

    public enum LimitType { min, max };

    public float robotX;
    public float robotZ;
    public float robotReachabilityLimit;
    public LimitType limitType;

    public override bool Passes(float placementX, float placementZ, float objectRadius){
        Vector2 placementPt = new Vector2(placementX, placementZ);
        Vector2 basePt = new Vector2(robotX, robotZ);
        float distance = Vector2.Distance(placementPt, basePt);
        if (limitType == LimitType.max){
            bool pass = distance < robotReachabilityLimit;
            return pass;
        }
        else{
            bool pass = distance > robotReachabilityLimit;
            return pass;
        }
    }
}

public class SurfaceObjectPlacer{
    
    public bool useFixedX = false;
    public bool useFixedZ = true;
    public float fixedX = 0.0f;
    public float fixedZ = -0.25f;
    public GameObject plane;
    private System.Random randomiser = new System.Random();
    private ReachabilityConstraint minRobotReachability;
    private ReachabilityConstraint maxRobotReachability;
    private ReachabilityConstraint visibilityConstraint;
    private int maxPlacementTries = 100;
    private float edgeMargin = 0.1f;
    private float stackingMargin = 0.001f;
    private List<PlacementConstraint> collisionConstraints = new List<PlacementConstraint>();


    public SurfaceObjectPlacer(
        GameObject plane,
        ReachabilityConstraint minRobotReachability,
        ReachabilityConstraint maxRobotReachability,
        ReachabilityConstraint visibilityConstraint,
        int maxPlacementTries){
        
            this.plane = plane;
            this.minRobotReachability = minRobotReachability;
            this.maxRobotReachability = maxRobotReachability;
            this.visibilityConstraint = visibilityConstraint;
            this.maxPlacementTries = maxPlacementTries;
    }


    public void IterationStart(){
        collisionConstraints = new List<PlacementConstraint>();
        plane = GameObject.Find("Table");
    }

    public (bool, List<PlacementConstraint>) PlaceObject(GameObject obj, List<PlacementConstraint> pre_constraints, bool respectMaxRobotReachability, float yOffset){
        Debug.Log("Place object at yOffset:"+yOffset);
        Bounds planeBounds = plane.GetComponent<Renderer>().bounds;
        if (obj.activeInHierarchy){
            // try to sample a valid point
            Bounds objBounds = obj.GetComponent<Renderer>().bounds;
            float radius = 0;
            float heightAbovePlane = planeBounds.extents.y+objBounds.extents.y+stackingMargin;

            List<PlacementConstraint> constraints = GetAllConstraints(respectMaxRobotReachability);
            constraints.AddRange(pre_constraints);
            Vector3? point = SampleValidGlobalPointOnPlane(radius, constraints, planeBounds, respectMaxRobotReachability);

            if (point.HasValue){
                // place object
                Vector3 foundPoint = point ?? Vector3.zero;
                obj.transform.position = new Vector3(foundPoint.x, foundPoint.y+heightAbovePlane+yOffset, foundPoint.z);

                // update constraints so subsequently placed object cannot collide with this one
                CollisionConstraint newConstraint = new CollisionConstraint();
                newConstraint.x = foundPoint.x;
                newConstraint.z = foundPoint.z;
                newConstraint.radius = radius;
                collisionConstraints.Add(newConstraint);
            }
            else{
                return (false, collisionConstraints);
            }
        }
        return (true, collisionConstraints);
    }

    // PRIVATE HELPERS

    private Vector3? SampleValidGlobalPointOnPlane(float objectRadius, List<PlacementConstraint> constraints, Bounds planeBounds, bool respectMaxRobotReachability){
        // return a valid point and if not found one it return null 
        int tries = 0;

        while (tries < maxPlacementTries)
        {
            Vector3 point = SampleGlobalPointOnPlane(objectRadius, planeBounds, respectMaxRobotReachability);
            bool valid = PassesConstraints(point, objectRadius, constraints);
            if (valid) { return point; }

            tries += 1;
        }
        return null;
    }

    private List<PlacementConstraint> GetAllConstraints(bool respectMaxRobotReachability){
        // return a list of all the constraints: combination of permanent constraint and additional constraint like the maxReachabilityConstraint or the 
        // collision constraint 
        List<PlacementConstraint> allConstraints = new List<PlacementConstraint>();
        allConstraints.AddRange(collisionConstraints);
        allConstraints.Add(minRobotReachability);
        if (respectMaxRobotReachability){
            allConstraints.Add(maxRobotReachability);
        }
        allConstraints.Add(visibilityConstraint);
        return allConstraints;
    }

    private Vector3 SampleGlobalPointOnPlane(float minEdgeDistance, Bounds planeBounds, bool respectMaxRobotReachability){
        Rect planePlacementZone = PlanePlacementZone(planeBounds, edgeMargin*minEdgeDistance);
        if (respectMaxRobotReachability){
            Rect withinMaxReachZone = MaxReachabilityPlacementZone(maxRobotReachability);
            planePlacementZone = Intersection(planePlacementZone, withinMaxReachZone);
        }

        Vector2 randomPlanePoint = RandomPointInRect(planePlacementZone);
        Vector3 globalPt = new Vector3(randomPlanePoint.x, planeBounds.center.y, randomPlanePoint.y);
        return globalPt;
    }


    private static Rect MaxReachabilityPlacementZone(ReachabilityConstraint maxRobotReachability){
        float x = maxRobotReachability.robotX - maxRobotReachability.robotReachabilityLimit;
        float z = maxRobotReachability.robotZ - maxRobotReachability.robotReachabilityLimit;
        float size = maxRobotReachability.robotReachabilityLimit * 2;
        return new Rect(x, z, size, size);
    }

    private static Rect PlanePlacementZone(Bounds planeBounds, float minEdgeDistance){
        float x = planeBounds.center.x - planeBounds.extents.x + minEdgeDistance;
        float z = planeBounds.center.z - planeBounds.extents.z + minEdgeDistance;
        float dx = (planeBounds.extents.x - minEdgeDistance) * 2;
        float dz = (planeBounds.extents.z - minEdgeDistance) * 2;
        return new Rect(x, z, dx, dz);
    }

    private Vector2 RandomPointInRect(Rect rect){
        float x, z;
        if (useFixedX) x = fixedX;
        else x = (float)randomiser.NextDouble() * rect.width + rect.xMin;
        
        if (useFixedZ) z = fixedZ;
        else z = (float)randomiser.NextDouble() * rect.height + rect.yMin;
        
        return new Vector2(x, z);
    }

    private static Rect Intersection(Rect rectA, Rect rectB){
        float minX = Mathf.Max(rectA.xMin, rectB.xMin);
        float maxX = Mathf.Min(rectA.xMax, rectB.xMax);
        float minY = Mathf.Max(rectA.yMin, rectB.yMin);
        float maxY = Mathf.Min(rectA.yMax, rectB.yMax);

        bool xValid = (minX < maxX);
        bool yValid = (minY < maxY);
        bool valid = (xValid && yValid);
        if (!valid){
            throw new Exception("Rectangles have no intersection!");
        }

        return new Rect(minX, minY, maxX - minX, maxY - minY);
    }

    private static bool PassesConstraints(Vector3 point, float objectRadius, List<PlacementConstraint> constraints)
    {
        /* Checks if sampled point on plane passes all provided constraints. */

        foreach (PlacementConstraint constraint in constraints)
        {
            bool pass = constraint.Passes(point.x, point.z, objectRadius);
            if (!pass) { return false; }
        }
        return true;
    }

}
