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
using System.IO;
using UnityEngine;

public class CaptureCameraScript : MonoBehaviour{
    
    public bool captureDepth = true;
    public int outputImageWidth = 800;
    public int outputImageHeight = 600;
    public float depthLevel = 0.5f;
    public string camera_name;
    
    private Material material;
    private RenderTexture renderTexture;
    private RenderTexture renderTextureRGB;
    private RenderTexture renderTextureD;
    private Shader depthShader;
    private Camera cam;
    private Texture2D texRGB;
    private Texture2D texD;
    
    private WaitForEndOfFrame frameEnd = new WaitForEndOfFrame();
    
    void Start(){
        // Create render textures
        renderTexture = new RenderTexture(outputImageWidth, outputImageHeight, 24, UnityEngine.Experimental.Rendering.GraphicsFormat.R8G8B8A8_SRGB);
        renderTexture.Create();
        renderTextureRGB = new RenderTexture(outputImageWidth, outputImageHeight, 24, UnityEngine.Experimental.Rendering.GraphicsFormat.R8G8B8A8_SRGB);
        renderTextureRGB.Create();
        renderTextureD = new RenderTexture(outputImageWidth, outputImageHeight, 24, UnityEngine.Experimental.Rendering.GraphicsFormat.R8G8B8A8_SRGB);
        renderTextureD.Create();
        
        // Create 2D textures
        texRGB = new Texture2D(outputImageWidth, outputImageHeight);
        texD = new Texture2D(outputImageWidth, outputImageHeight, TextureFormat.RGB24, false);
        
        // Set buffers and texture mode
        cam = gameObject.GetComponent<Camera>();
        cam.SetTargetBuffers(renderTexture.colorBuffer, renderTexture.depthBuffer);
        cam.depthTextureMode = DepthTextureMode.Depth;
        
        // Retrieve the depth shader and create the depth rendering material
        depthShader = Shader.Find("Custom/LinearDepthShader");
        material = new Material(depthShader);
        
        // Disable the snapshot camera
        cam.enabled = false;
    }
    
    public void AlignToMainCamera(){
        cam.transform.eulerAngles = Camera.main.transform.eulerAngles;
        cam.transform.position = Camera.main.transform.position;
    }
    
    public void CaptureRGBD(string path){
        StartCoroutine(CaptureRoutine(path));
    }
    
    public IEnumerator CaptureRoutine(string path){
        // Enable the snapshot camera and wait for the end of the frame
        cam.enabled = true;
        
        float stored_timescale = Time.timeScale;
        
        Time.timeScale = 0.0f;
        yield return frameEnd;
        
        // Capture an RGB image from this camera
        cam.Render();
        texRGB.ReadPixels(new Rect(0, 0, outputImageWidth, outputImageHeight), 0, 0);
        texRGB.Apply();
        byte[] bytesRGB = texRGB.EncodeToPNG();
        string pathRGB = path+camera_name+"_RGB.png";
        File.WriteAllBytes(pathRGB, bytesRGB);
        Debug.Log("Saved RGB image to: "+pathRGB);
        
        // Capture a depth image from this camera
        Graphics.Blit(renderTexture, renderTextureD, material);
        texD.ReadPixels(new Rect(0, 0, outputImageWidth, outputImageHeight), 0, 0);
        texD.Apply();
        byte[] bytes = texD.EncodeToPNG();
        string pathD = path+camera_name+"_D.png";
        File.WriteAllBytes(pathD, bytes);
        Debug.Log("Saved D image to: "+pathD);
        
        // Disable the snapshot camera
        cam.enabled = false;
        
        yield return frameEnd;
        Time.timeScale = 1.0f;
        Debug.Log("time scale restored to: "+Time.timeScale);
    }
    
}
