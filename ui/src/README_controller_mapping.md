# Controller Mapping for ComfyUI_F

This document explains how to use the controller mapping feature in ComfyUI_F.

## Controller Mapping Types

The system supports two main types of controller mappings:

1. **Axis Mapping**: Maps controller analog inputs (joysticks, triggers) to numerical values
2. **Button Mapping**: Maps controller buttons to values, with three different modes:
   - **Momentary**: Value changes while button is held (pressed/released)
   - **Toggle**: Value toggles between two states with each press
   - **Series**: Cycle through a list of predefined values with each press

## Setting Up Controller Mappings

1. Select a node and field you want to control via the control panel
2. Click on the "Map" button next to the field
3. Choose the mapping type and configure it:

### Axis Mapping
- Select the axis number (joystick or trigger)
- Adjust multiplier to control sensitivity
- Set min/max values to constrain the output range

### Button Mapping
- Select the button number
- Choose a mode:
  - **Momentary**: Define values for when button is pressed/released
  - **Toggle**: Define values to toggle between
  - **Series**: Define a list of values to cycle through

### Series Mode Configuration
Series mode allows you to cycle through a list of values with button presses:

1. Select "Series" mode
2. Add values to cycle through in the list
3. Optionally set a second button to cycle backwards through the list

## Example Use Cases

### Axis Mapping
- Control seed values with a joystick
- Adjust CFG Scale with a trigger
- Control denoising strength with analog input

### Button Mapping (Momentary)
- Toggle sampling methods on/off
- Run/stop generation with a button press

### Button Mapping (Toggle)
- Switch between different model types
- Toggle clip skip settings

### Button Mapping (Series)
- Cycle through different prompt templates
- Cycle through sampler options (Euler, Euler a, DPM++)
- Cycle through different step counts (20, 30, 50)
- Cycle through checkpoint models

## Tips for Using Controller Mappings

- For numerical values, Axis mapping provides smooth control
- For cycling through discrete options, use Button with Series mode
- You can map multiple fields to different controller inputs
- Series mode is perfect for switching between common presets

## Troubleshooting

If your controller isn't being detected:
1. Make sure it's connected before opening the page
2. Try pressing a button on the controller to wake it up
3. Use the "Refresh Controllers" button in the mapping dialog
4. Check browser console for any errors

If button presses aren't being detected:
1. Make sure you've assigned the correct button indices
2. Use the "Detect" button to automatically find the right indices
3. Check that the controller is still connected 