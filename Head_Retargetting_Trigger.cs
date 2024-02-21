using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Valve.VR;

public class Head_Retargetting_Trigger : MonoBehaviour
{
  public SteamVR_Action_Boolean headRetargettingAction;
  public string headRetargettingId;
  public SteamVR_Input_Sources actionSource;
  private bool currentState = false;
  public bool displayRetargetting = false;

    [SerializeField] private SG.SG_GestureLayer gestureLayer;
    [SerializeField] private SG.SG_BasicGesture headRetargetGesture;

    [SerializeField] private AudioSource audioSource;

    // Start is called before the first frame update
    void Start()
  {
        audioSource = GetComponent<AudioSource>();
    }

    // Update is called once per frame
    void Update()
    {
        if (headRetargettingAction.GetStateDown(actionSource))
        {
            McRtc.Client.SendCheckboxRequest(headRetargettingId, currentState);
            currentState = !currentState;
            displayRetargetting = currentState;
        }
        if (headRetargetGesture != null) { 
            if (headRetargetGesture.GestureStopped == true)
            {
                McRtc.Client.SendCheckboxRequest(headRetargettingId, currentState);
                currentState = !currentState;
                displayRetargetting = currentState;
                if (audioSource.clip != null)
                {
                    audioSource.Play();
                }
            }
        }
    }
}

