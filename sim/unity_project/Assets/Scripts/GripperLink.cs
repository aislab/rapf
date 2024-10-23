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

public class GripperLink : MonoBehaviour{
    
    private bool supportPhysics = true;
    private GameObject target;
    Vector3 previousTargetPosition = Vector3.zero;
    private Rigidbody rb;
        
    public void setup(){
        // find the gripper base
        target = GameObject.Find("GripperFollowTarget");
        rb = GetComponent<Rigidbody>();
        rb.useGravity = false;
        rb.isKinematic = true;
    }

    // Update is called once per frame
    void Update(){

        // Move our position a step closer to the target.
        float positionStep = 100.0f*Time.deltaTime; // calculate distance to move
        float rotationStep = 100.0f*Time.deltaTime;
        
        if (supportPhysics){
            Vector3 pos = Vector3.MoveTowards(transform.position, target.transform.position, positionStep);
            Quaternion rot = Quaternion.RotateTowards(transform.rotation, target.transform.rotation, rotationStep);
            rb.MovePosition(pos);
            rb.MoveRotation(rot);
        }
        else{
            transform.position = Vector3.MoveTowards(transform.position, target.transform.position, positionStep);
            transform.rotation = Quaternion.RotateTowards(transform.rotation, target.transform.rotation, rotationStep);
        }
    }
    
}
