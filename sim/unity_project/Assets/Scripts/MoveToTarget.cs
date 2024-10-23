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

public class MoveToTarget : MonoBehaviour{
    
    private Rigidbody rb;
    private Vector3 targetPos;
    private Quaternion targetRot;
    private Queue<Vector3> targetPosQueue;
    private Queue<Quaternion> targetRotQueue;
    private bool gravity;
    private bool kinematic;
    private float mass;
    private int maxFrames = 1000;
    private int frame = 0;

    public void SetTarget(Vector3 pos, Quaternion rot){
        rb = GetComponent<Rigidbody>();
        targetPos = pos;
        targetRot = rot;
        gravity = rb.useGravity;
        kinematic = rb.isKinematic;
        rb.useGravity = false;
        rb.isKinematic = true;
        mass = rb.mass;
        rb.mass = 100f;
        Debug.Log("MoveToTarget SetTarget: "+targetPos+" starts at: "+transform.position);
    }
    
    public void AddTarget(Vector3 pos, Quaternion rot){
        if (targetPosQueue==null){
            targetPosQueue = new Queue<Vector3>();
            targetRotQueue = new Queue<Quaternion>();
        }
        targetPosQueue.Enqueue(pos);
        targetRotQueue.Enqueue(rot);
    }
    
    void FixedUpdate(){
        float positionStep =  0.01f;
        float rotationStep = 1.0f;
        Vector3 pos = Vector3.MoveTowards(transform.position, targetPos, positionStep);
        Quaternion rot = Quaternion.RotateTowards(transform.rotation, targetRot, rotationStep);
        rb.MovePosition(pos);
        rb.MoveRotation(rot);
        if ((Vector3.Distance(transform.position,targetPos) < 0.001f && Quaternion.Angle(transform.rotation,targetRot)<0.001f) || frame==maxFrames){
            if (frame==maxFrames)
                Debug.Log("MoveToTarget hit frame limit...");
            else
                Debug.Log("MoveToTarget reached target: "+targetPos+" "+targetRot);
            
            if (targetPosQueue==null || targetPosQueue.Count==0){
                rb.useGravity = gravity;
                rb.isKinematic = kinematic;
                rb.mass = mass;
                rb.velocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
                Destroy(this);
            }
            else{
                targetPos = targetPosQueue.Dequeue();
                targetRot = targetRotQueue.Dequeue();
                frame = 0;
            }
        }
        frame += 1;
    }
    
}
