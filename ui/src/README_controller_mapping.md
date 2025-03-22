# Controller Mapping for comfystream

This document explains how to use the controller mapping feature in comfystream. The system supports mapping from multiple input types:

- Game controllers/gamepads (Xbox, PlayStation, etc.)
- Keyboard keys
- Mouse buttons and movements

## Controller Mapping Types

The system supports multiple mapping types across all input devices:

1. **Axis Mapping**: Maps analog inputs to numerical values
   - Gamepad joysticks and triggers
   - Mouse X/Y movements
   - Mouse wheel

2. **Button Mapping**: Maps buttons/keys to values, with four different modes:
   - Gamepad buttons
   - Keyboard keys
   - Mouse buttons (left, middle, right)
   
   Available modes:
   - **Momentary**: Value changes while button is held (pressed/released)
   - **Toggle**: Value toggles between two states with each press
   - **Series**: Cycle through a list of predefined values with each press
   - **Increment**: Increment/decrement a numerical value by a defined step

## Setting Up Controller Mappings

1. Select a node and field you want to control via the control panel
2. Click on the "Map" button next to the field
3. Choose the input type you want to use (gamepad, keyboard, or mouse)
4. Select the mapping type and configure it according to your selected input:

### Axis Mapping
For gamepad analog inputs or mouse movements:
- For gamepad: Select the axis number (joystick or trigger)
- For mouse: Choose X-axis, Y-axis, or wheel
- Adjust multiplier to control sensitivity
- Set min/max values to constrain the output range

### Button Mapping
For gamepad buttons, keyboard keys, or mouse buttons:
- For gamepad: Select the button number
- For keyboard: Select the key to use
- For mouse: Select which mouse button (left, middle, right)
- Choose a mode:
  - **Momentary**: Define values for when button is pressed/released
  - **Toggle**: Define values to toggle between
  - **Series**: Define a list of values to cycle through
  - **Increment**: Define the increment step and optional min/max bounds

### Series Mode Configuration
Series mode allows you to cycle through a list of values with button presses:

1. Select "Series" mode
2. Add values to cycle through in the list
3. Optionally set a second button to cycle backwards through the list

### Increment Mode Configuration
Increment mode allows you to increase or decrease numerical values by a fixed amount:

1. Select "Increment" mode
2. Set the increment step (amount to change per button press)
3. Optionally set a minimum and maximum value to constrain the range
4. Configure a decrement button to decrease the value

Press-and-hold functionality is supported:
- Press and release for a single increment/decrement
- Press and hold to continuously increment/decrement values
- Releasing the button stops the continuous adjustment

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

### Button Mapping (Increment)
- Fine-tune numerical parameters like CFG scale or denoise strength
- Increase/decrease step count in precise increments
- Adjust batch size up or down with button presses
- Precisely control slider values with button presses

## Tips for Using Controller Mappings

- For numerical values, Axis mapping provides smooth control
- For cycling through discrete options, use Button with Series mode
- For precise adjustments to numerical values, use Increment mode
- Use Increment mode when you need fine-grained control over parameters
- You can map multiple fields to different controller inputs
- Series mode is perfect for switching between common presets
- Increment mode works best for parameters that need small adjustments

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

Issues with increment mode:
1. If values aren't incrementing/decrementing correctly, verify both buttons are correctly mapped
2. Check that min/max values are appropriate for the parameter (allow room for increments)
3. For fine-tuning, use smaller increment steps (0.1 or 0.01)
4. For larger changes, increase the increment step value 