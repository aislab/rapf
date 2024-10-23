/*
Copyright 2024 Autonomous Intelligence and Systems (AIS) Lab, Shinshu University

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
using System;

public class GraspAngleScript : MonoBehaviour{
    
    private float rotationStep;
    private float[] roundGrasp;
    private float[] roundGraspSymmetrical;
    private float[] squareGrasp = new float[] {0,90,180,270};
    private float[] squareGraspSymmetrical = new float[] {0,90};
    
    public enum myEnum {round,square,custom};
    public bool radialSymmetryX = false;
    public bool radialSymmetryY = false;
    public bool radialSymmetryZ = false;
    public GraspAngleScript.myEnum XAxisGraspPattern = myEnum.square;
    public GraspAngleScript.myEnum YAxisGraspPattern = myEnum.square;
    public GraspAngleScript.myEnum ZAxisGraspPattern = myEnum.square;
    
    public float[] customAnglesX;
    public float[] customAnglesY;
    public float[] customAnglesZ;
    
    
    // Start is called before the first frame update
    void Start(){
        if (XAxisGraspPattern == myEnum.round || YAxisGraspPattern == myEnum.round || ZAxisGraspPattern == myEnum.round){
            rotationStep = GameObject.Find("AffordanceChecker").GetComponent<AffordanceCheck>().rotationStep;
            if (180%rotationStep!=0){
                Debug.Log("Bad rotationStep setting: rotationStep should cleanly divide 180");
                Debug.Break();
            }
            int n_angles = (int)(360/rotationStep);
            //Debug.Log("rotationStep:"+rotationStep+" n_angles:"+n_angles);
            roundGrasp = new float[n_angles];
            for (int i=0;i<n_angles;i++)
                roundGrasp[i] = i*rotationStep;
            roundGraspSymmetrical = new ArraySegment<float>(roundGrasp, 0, n_angles/2).ToArray();
        }
    }
    
    private float[] GetRoundAngles(bool symmetrical){
        if (symmetrical) return roundGraspSymmetrical;
        else return roundGrasp;
    }
    
    private float[] AddCurrentAngle(float[] graspAngles, float angle){
        float[] angles = (float[]) graspAngles.Clone();
        for (int i=0; i<graspAngles.Length; i++){
            angles[i] = -(angles[i]+angle)%360;
            if (Math.Abs(angles[i]) >= 180) angles[i] -= Math.Sign(angles[i])*360;
        }
        return angles;
    }
    
    public (float[] angles, bool symmetry) GetGraspAngles(){
        Quaternion q = this.gameObject.transform.rotation;
        Debug.Log(this.name+": x:"+q.x+" y:"+q.y+" z:"+q.z+" w:"+q.w);
        Vector3 xVec = new Vector3(1.0f,0.0f,0.0f);
        Vector3 yVec = new Vector3(0.0f,1.0f,0.0f);
        Vector3 zVec = new Vector3(0.0f,0.0f,1.0f);
        float x_upness = Math.Abs((q*xVec).y);
        float y_upness = Math.Abs((q*yVec).y);
        float z_upness = Math.Abs((q*zVec).y);
        Debug.Log(this.name+": x_upness:"+x_upness+" y_upness:"+y_upness+" z_upness:"+z_upness);
        float vx = 0;
        float vz = 0;
        if (y_upness >= x_upness && y_upness >= z_upness){
            Debug.Log("y up");
            if (YAxisGraspPattern == myEnum.round) return (GetRoundAngles(radialSymmetryY),radialSymmetryY);
            else{
                vx = (q*zVec).x;
                vz = (q*zVec).z;
                float angle = Mathf.Atan2(vz,vx)*Mathf.Rad2Deg;
                Debug.Log("vx,vz: ("+vx+","+vz+") angle: "+angle);
                if (YAxisGraspPattern == myEnum.square)return (AddCurrentAngle(radialSymmetryY?squareGraspSymmetrical:squareGrasp,angle),radialSymmetryY);
                else return (AddCurrentAngle(customAnglesY,angle),radialSymmetryY);
            }
        }
        if (x_upness >= y_upness && x_upness >= z_upness){
            Debug.Log("x up");
            if (XAxisGraspPattern == myEnum.round) return (GetRoundAngles(radialSymmetryX),radialSymmetryX);
            else{
                vx = (q*yVec).x;
                vz = (q*yVec).z;
                float angle = Mathf.Atan2(vz,vx)*Mathf.Rad2Deg;
                Debug.Log("vx,vz: ("+vx+","+vz+") angle: "+angle);
                if (XAxisGraspPattern == myEnum.square)return (AddCurrentAngle(radialSymmetryX?squareGraspSymmetrical:squareGrasp,angle),radialSymmetryX);
                else return (AddCurrentAngle(customAnglesX,angle),radialSymmetryX);
            }
        }
        if (z_upness >= x_upness && z_upness >= y_upness){
            Debug.Log("z up");
            if (ZAxisGraspPattern == myEnum.round) return (GetRoundAngles(radialSymmetryZ),radialSymmetryZ);
            else{
                vx = (q*xVec).x;
                vz = (q*xVec).z;
                float angle = Mathf.Atan2(vz,vx)*Mathf.Rad2Deg;
                Debug.Log("vx,vz: ("+vx+","+vz+") angle: "+angle);
                if (ZAxisGraspPattern == myEnum.square)return (AddCurrentAngle(radialSymmetryZ?squareGraspSymmetrical:squareGrasp,angle),radialSymmetryZ);
                else return (AddCurrentAngle(customAnglesZ,angle),radialSymmetryZ);
            }
        }
        return (null,false);
    }
    
}
