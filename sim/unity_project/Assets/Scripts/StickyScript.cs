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

public class StickyScript : MonoBehaviour{
    
    public GameObject stickingTo;
    private GameObject ignore;
    private Rigidbody rb;
    private FixedJoint joint;
    private bool active = false;
    
    // Start is called before the first frame update
    void Start(){
        rb = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update(){
        if (stickingTo != null){
            float xrot = stickingTo.transform.eulerAngles.z;
            if (xrot>180)xrot-=360;
            if (Math.Abs(xrot)>5) Detach();
            float zrot = stickingTo.transform.eulerAngles.z;
            if (zrot>180)zrot-=360;
            if (Math.Abs(zrot)>5) Detach();
        }
    }
    
    void OnCollisionEnter(Collision collision){
        if (!active) return;
        if (collision.collider.gameObject.name != "4x1x1") return;
        if (stickingTo == null && collision.collider.gameObject != ignore){
            stickingTo = collision.collider.gameObject;
            joint = gameObject.AddComponent<FixedJoint>();
            joint.enablePreprocessing = true;
            joint.connectedBody=stickingTo.GetComponent<Rigidbody>();
        }
    }
    
    void OnCollisionStay(Collision collision){
        if (!active) return;
        if (collision.collider.gameObject.name != "4x1x1") return;
        if (stickingTo == null && collision.collider.gameObject != ignore){
            stickingTo = collision.collider.gameObject;
            joint = gameObject.AddComponent<FixedJoint>();
            joint.enablePreprocessing = true;
            joint.connectedBody=stickingTo.GetComponent<Rigidbody>();
        }
    }
    
    public void Detach(){
        ignore = stickingTo;
        if (joint != null){
            joint.connectedBody=null;
            Destroy(joint);
        }
        Debug.Log("Detach");
    }
    
    public void Forget(){
        ignore = null;
    }
    
    public void Activate(){
        active = true;
    }
    
}
