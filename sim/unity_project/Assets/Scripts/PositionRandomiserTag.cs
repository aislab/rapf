/*
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
using UnityEngine.Perception.Randomization.Randomizers;

[AddComponentMenu("Perception/RandomizerTags/PositionRandomiserTag")]
public class PositionRandomiserTag : RandomizerTag{
    
    public string randomisationGroup;
    public bool startUpright = false;
    public float presenceProbability = 1.0f;
    public float priority = 1;
    public bool mustBeReachable;
    public bool stackable;
    public bool doNotStackOnThisObject;
    public bool randomiseRotationY = false;
    public int rotationStepY = 180;
    public bool spread;
    public bool discretisedX;
    public float discretisationStepX = 0.1f;
    public bool discretisedZ;
    public float discretisationStepZ = 0.1f;
    public bool discretisationNoise = true;
    public bool placeAroundInitialPosition;
    public float rangeAroundInitialPosition;
    public bool placeInPlacementArea;
    public GameObject placementAreaObject;
    public Vector3 initialPosition;
    public Quaternion initialRotation;
    [HideInInspector] public float originalDrag;
    private System.Random randomiser = new System.Random();
    
    void Awake(){
        initialPosition = this.transform.position;
        initialRotation = this.transform.rotation;
        priority += 0.001f*(float)randomiser.NextDouble();
    }
}
