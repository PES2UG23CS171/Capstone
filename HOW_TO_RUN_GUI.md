# How to Run the GUI and Waveform Viewer Demo

Follow these instructions to run the updated application with the GUI and waveform viewer.

## 1. Open your Terminal
Make sure your terminal is opened inside the Capstone project folder:

## 2. Activate the Virtual Environment
*(If it's not already activated from earlier)*
```powershell
.\.venv_poc\Scripts\activate
```

## 3. Launch the Application
Start the main GUI by running:
```powershell
python -m app.main
```

---

## How to use the GUI during the demo:

1. **Verify live processing:** When the window opens, you will see the level meters responding to your microphone.
2. **Show the metrics:** Next to the "Suppression: ON" button, you should see the live **RTF** (Real-Time Factor) value updating in green text (e.g., `RTF: 0.0300 (33× headroom)`).
3. **Open the Waveform Viewer:** To show the offline analysis, click the new **"⚡ Proof of Concept"** button at the bottom of the window.
4. **Explain the plots:** The Waveform Viewer will open, generate the testing signals, and automatically plot the Clean, Noisy, and Filtered waveforms with the transient markers for you to showcase.

*Note: When you are finished, you can close the window, right-click the green microphone icon in your system tray (bottom right corner of Windows), and select "Quit" to fully exit the application.*

---

## 🎤 60-Second Demo Script

You can use this script when presenting the GUI to Panel:

1. *(Toggle the "Suppression: ON" button)* "We've integrated our proof-of-concept DSP filter directly into our Python GUI. As you can see by this live RTF number updating here, we are successfully processing audio at a fraction of the required real-time budget."
2. *(Clap or speak loudly into the mic)* "If I make a loud sound into the microphone, you can hear it being instantly attenuated in the output, with zero perceptible delay."
3. *(Click the "Proof of Concept" button to open the Waveform Viewer)* "To make the results easier to visualize, we built this offline processing viewer. It generates a synthetic signal with exactly five different transient noises, and stacks the waveforms so we can compare the clean, noisy, and filtered results."
4. *(Zoom into the 8.5s mark on the plots)* "As you can see, the dog bark and door slam are successfully suppressed. But importantly, look at the 8.5-second mark. This is a simulated 'P' plosive, just like in human speech. The filter successfully preserved it without triggering suppression."
5. *(Click Play Noisy, then Play Filtered)* "Listen to the difference in the background noise and the transients..." *(Let it play)*
6. *(Point to the stats at the bottom)* "And finally, looking at our metrics, we achieved a Real-Time Factor (RTF) of around 0.03. This means our pipeline runs about 30 times faster than real-time, leaving us with plenty of headroom for when we drop the heavy Mamba and DeepFIR neural networks into this exact same pipeline in Phase 2. The feasibility is a definitive Pass."
