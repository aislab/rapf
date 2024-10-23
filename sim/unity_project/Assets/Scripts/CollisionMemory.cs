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

public class CollisionMemory : MonoBehaviour{
    
    public bool has_collided = false;
    public List<Collider> colliderList = new List<Collider>();
    public List<Collider> ignoreList = new List<Collider>();
    
    void RecordCollision(Collider other){
        if (!ignoreList.Contains(other)){
            if (!colliderList.Contains(other)){
                has_collided = true;
                colliderList.Add(other);
            }
        }
    }
    
    public void AddIgnore(Collider other){
        ignoreList.Add(other);
    }
    
    public void ClearCollisionMemory(){
        has_collided = false;
        colliderList.Clear();
    }
    
    void OnTriggerEnter(Collider other){
        RecordCollision(other);
    }
    
    void OnCollisionEnter(Collision collision){
        RecordCollision(collision.collider);
    }
    
    void OnTriggerStay(Collider other){
        RecordCollision(other);
    }
    
    void OnCollisionStay(Collision collision){
        RecordCollision(collision.collider);
    }
    
}
