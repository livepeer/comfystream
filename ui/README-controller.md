# Controller Mapping for ComfyUI

This feature allows you to map gamepad/controller inputs to ComfyUI node parameters, making it easy to adjust values in real-time using a controller.

## Usage

1. Connect a gamepad or controller to your computer
2. Open the control panel interface in ComfyUI
3. Select a node and parameter you want to control
4. Click the "Map" button next to the parameter input
5. Choose the mapping type and configure it
6. Save the mapping
7. Enable "Auto-Update" to make your controller inputs affect the parameter in real-time

## Mapping Types

### Axis Mapping

Maps an analog axis (joystick, trigger) to a numeric parameter:

- **Axis**: The axis index (0-20) to use
- **Multiplier**: Scale factor to apply to the axis value
- **Min/Max Value**: Optional range overrides

### Button Mapping

Maps a button press to a specific value:

- **Button**: The button index (0-20) to use
- **Toggle Mode**: When enabled, button acts as a toggle between two values
- **Value When Pressed**: Value to set when button is pressed
- **Value When Released**: Value to set when button is released (or toggle off value)

### Prompt List Mapping

Maps buttons to cycle through the available prompts:

- **Next Prompt Button**: Button to select the next prompt
- **Previous Prompt Button**: Button to select the previous prompt

## Detection Mode

For easier setup, you can click "Detect" and:
- For axis mapping: Move the joystick or trigger you want to use
- For button mapping: Press the button you want to map
- For prompt list: Press first button (Next) then second button (Prev)

## Technical Details

The controller mapping system is built using:

- **useController**: Hook to detect controllers and track button/axis states
- **useControllerMapping**: Hook to manage and persist controller mappings
- **useControllerInput**: Hook to apply controller input based on mapping

Mappings are saved in localStorage and restored when the interface loads. 