"use client";

import React, { useState, useEffect, useRef } from "react";
import { usePeerContext } from "@/context/peer-context";
import { usePrompt } from "./settings";
import { useControllerInput } from "@/hooks/use-controller-input";
import { useControllerMapping } from "@/hooks/use-controller-mapping";
import { ControllerMappingButton } from "./controller-mapping";
import { AxisMapping, ButtonMapping } from "@/types/controller";

type InputValue = string | number | boolean;

interface InputInfo {
  value: InputValue;
  type: string;
  min?: number;
  max?: number;
  widget?: string;
}

interface NodeInfo {
  class_type: string;
  inputs: Record<string, InputInfo>;
}

interface ControlPanelProps {
  panelState: {
    nodeId: string;
    fieldName: string;
    value: string;
    isAutoUpdateEnabled: boolean;
  };
  onStateChange: (
    state: Partial<{
      nodeId: string;
      fieldName: string;
      value: string;
      isAutoUpdateEnabled: boolean;
    }>,
  ) => void;
}

// Add a shared debounce mechanism at the top of the file (after imports)
const UPDATE_QUEUE: Record<string, any> = {};
let updateTimeoutId: NodeJS.Timeout | null = null;
const DEBOUNCE_TIME_MS = 50; // Default debounce for button controls
const AXIS_DEBOUNCE_TIME_MS = 0; // No debounce for axis controls to maintain fluidity

// Process all pending updates from multiple control panels
const processUpdateQueue = (controlChannel: RTCDataChannel | null, currentPrompts: any[], setCurrentPrompts: (prompts: any[]) => void) => {
  if (!controlChannel || Object.keys(UPDATE_QUEUE).length === 0) return;
  
  console.log("Processing queued updates:", UPDATE_QUEUE);
  
  // Create a deep copy of prompts to apply all updates
  const updatedPrompts = JSON.parse(JSON.stringify(currentPrompts));
  
  // Apply all queued updates
  Object.entries(UPDATE_QUEUE).forEach(([key, update]) => {
    const [promptIdx, nodeId, fieldName] = key.split('|');
    const idx = parseInt(promptIdx);
    
    if (updatedPrompts[idx] && updatedPrompts[idx][nodeId]?.inputs) {
      updatedPrompts[idx][nodeId].inputs[fieldName] = update.value;
      console.log(`Applied update to prompt ${idx}, node ${nodeId}, field ${fieldName}:`, update.value);
    }
  });
  
  // Send the consolidated update to ComfyUI
  const message = JSON.stringify({
    type: "update_prompts",
    prompts: updatedPrompts,
  });
  
  console.log("Sending consolidated updates to ComfyUI");
  controlChannel.send(message);
  
  // Update local prompts
  setCurrentPrompts(updatedPrompts);
  
  // Clear the queue
  Object.keys(UPDATE_QUEUE).forEach(key => delete UPDATE_QUEUE[key]);
};

const InputControl = ({
  input,
  value,
  onChange,
}: {
  input: InputInfo;
  value: string;
  onChange: (value: string) => void;
}) => {
  // Add a state to track recent controller updates for visual feedback
  const [isControllerUpdated, setIsControllerUpdated] = useState(false);
  const controllerUpdateTimer = useRef<NodeJS.Timeout | null>(null);
  
  // When the value changes, show controller update indicator
  useEffect(() => {
    // Check if this is from initial mount or actual update
    if (value !== undefined) {
      setIsControllerUpdated(true);
      
      // Clear any existing timer
      if (controllerUpdateTimer.current) {
        clearTimeout(controllerUpdateTimer.current);
      }
      
      // Set a timer to remove the indicator
      controllerUpdateTimer.current = setTimeout(() => {
        setIsControllerUpdated(false);
      }, 300); // Flash for 300ms
    }
    
    return () => {
      if (controllerUpdateTimer.current) {
        clearTimeout(controllerUpdateTimer.current);
      }
    };
  }, [value]);
  
  // Helper to apply controller update styles
  const controllerUpdateStyle = isControllerUpdated 
    ? { boxShadow: '0 0 0 2px rgba(59, 130, 246, 0.5)', transition: 'box-shadow 0.2s ease-out' } 
    : { transition: 'box-shadow 0.2s ease-out' };
  
  if (input.widget === "combo") {
    return (
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="p-2 border rounded w-full"
        style={controllerUpdateStyle}
      >
        {Array.isArray(input.value) &&
          input.value.map((option: string) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
      </select>
    );
  }

  // Convert type to lowercase for consistent comparison
  const inputType = input.type.toLowerCase();

  switch (inputType) {
    case "boolean":
      return (
        <div style={{ display: 'inline-block', ...controllerUpdateStyle }}>
          <input
            type="checkbox"
            checked={value === "true"}
            onChange={(e) => onChange(e.target.checked.toString())}
            className="w-5 h-5"
          />
        </div>
      );
    case "number":
    case "float":
    case "int":
      return (
        <input
          type="number"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          min={input.min}
          max={input.max}
          step={
            inputType === "float" ? "0.01" : inputType === "int" ? "1" : "any"
          }
          className="p-2 border rounded w-32"
          style={controllerUpdateStyle}
        />
      );
    case "string":
      return (
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="p-2 border rounded w-full"
          style={controllerUpdateStyle}
        />
      );
    default:
      console.warn(`Unhandled input type: ${input.type}`); // Debug log
      return (
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="p-2 border rounded w-full"
          style={controllerUpdateStyle}
        />
      );
  }
};

export const ControlPanel = ({
  panelState,
  onStateChange,
}: ControlPanelProps) => {
  const { controlChannel } = usePeerContext();
  const { currentPrompts, setCurrentPrompts } = usePrompt();
  const [availableNodes, setAvailableNodes] = useState<
    Record<string, NodeInfo>[]
  >([{}]);
  const [promptIdxToUpdate, setPromptIdxToUpdate] = useState<number>(0);

  // Add ref to track last sent value and timeout
  const lastSentValueRef = React.useRef<{
    nodeId: string;
    fieldName: string;
    value: InputValue;
  } | null>(null);
  const updateTimeoutRef = React.useRef<NodeJS.Timeout | null>(null);

  // Cleanup function for the timeout
  useEffect(() => {
    return () => {
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
    };
  }, []);

  // Request available nodes when control channel is established
  useEffect(() => {
    if (controlChannel) {
      controlChannel.send(JSON.stringify({ type: "get_nodes" }));

      controlChannel.addEventListener("message", (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "nodes_info") {
            setAvailableNodes(data.nodes);
          } else if (data.type === "prompts_updated") {
            if (!data.success) {
              console.error("[ControlPanel] Failed to update prompt");
            }
          }
        } catch (error) {
          console.error("[ControlPanel] Error parsing node info:", error);
        }
      });
    }
  }, [controlChannel]);

  const handleValueChange = (newValue: string) => {
    const currentInput =
      panelState.nodeId && panelState.fieldName
        ? availableNodes[promptIdxToUpdate][panelState.nodeId]?.inputs[
            panelState.fieldName
          ]
        : null;

    if (currentInput) {
      // Validate against min/max if they exist for number types
      if (currentInput.type === "number") {
        const numValue = parseFloat(newValue);
        if (!isNaN(numValue)) {
          if (currentInput.min !== undefined && numValue < currentInput.min)
            return;
          if (currentInput.max !== undefined && numValue > currentInput.max)
            return;
        }
      }
    }

    onStateChange({ value: newValue });
  };

  // Modify the effect that sends updates with debouncing
  useEffect(() => {
    const currentInput =
      panelState.nodeId && panelState.fieldName
        ? availableNodes[promptIdxToUpdate][panelState.nodeId]?.inputs[
            panelState.fieldName
          ]
        : null;
    if (!currentInput || !currentPrompts) return;

    let isValidValue = true;
    let processedValue: InputValue = panelState.value;

    // Validate and process value based on type
    switch (currentInput.type.toLowerCase()) {
      case "number":
        isValidValue =
          /^-?\d*\.?\d*$/.test(panelState.value) && panelState.value !== "";
        processedValue = parseFloat(panelState.value);
        break;
      case "boolean":
        isValidValue =
          panelState.value === "true" || panelState.value === "false";
        processedValue = panelState.value === "true";
        break;
      case "string":
        // String can be empty, so always valid
        processedValue = panelState.value;
        break;
      default:
        if (currentInput.widget === "combo") {
          isValidValue = panelState.value !== "";
          processedValue = panelState.value;
        } else {
          isValidValue = panelState.value !== "";
          processedValue = panelState.value;
        }
    }

    const hasRequiredFields =
      panelState.nodeId.trim() !== "" && panelState.fieldName.trim() !== "";

    // Check if the value has actually changed
    const lastSent = lastSentValueRef.current;
    const hasValueChanged =
      !lastSent ||
      lastSent.nodeId !== panelState.nodeId ||
      lastSent.fieldName !== panelState.fieldName ||
      lastSent.value !== processedValue;

    if (
      controlChannel &&
      panelState.isAutoUpdateEnabled &&
      isValidValue &&
      hasRequiredFields &&
      hasValueChanged
    ) {
      // Clear any existing timeout
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }

      // Determine if this is an axis control - assuming UI changes are not from axis controls
      // Use a shorter delay for numbers to remain responsive, longer for text
      const inputDelay = currentInput.type.toLowerCase() === "number" ? 100 : 300;

      // Set a new timeout for the update
      updateTimeoutRef.current = setTimeout(
        () => {
          // Update last sent value
          lastSentValueRef.current = {
            nodeId: panelState.nodeId,
            fieldName: panelState.fieldName,
            value: processedValue,
          };
          
          // Create a unique key for this update
          const updateKey = `${promptIdxToUpdate}|${panelState.nodeId}|${panelState.fieldName}`;
          
          // Add to the update queue with the processed value
          UPDATE_QUEUE[updateKey] = {
            value: processedValue,
            timestamp: Date.now(),
          };
          
          // Clear any existing global timeout
          if (updateTimeoutId) {
            clearTimeout(updateTimeoutId);
          }
          
          // For UI updates we'll use the regular debounce time since these aren't axis inputs
          updateTimeoutId = setTimeout(() => {
            processUpdateQueue(controlChannel, currentPrompts, setCurrentPrompts);
            updateTimeoutId = null;
          }, DEBOUNCE_TIME_MS);
        },
        inputDelay
      );
    }
  }, [
    panelState.value,
    panelState.nodeId,
    panelState.fieldName,
    panelState.isAutoUpdateEnabled,
    controlChannel,
    currentPrompts,
    setCurrentPrompts,
    availableNodes,
    promptIdxToUpdate,
  ]);

  const toggleAutoUpdate = () => {
    onStateChange({ isAutoUpdateEnabled: !panelState.isAutoUpdateEnabled });
  };

  // Modified to handle initial values better
  const getInitialValue = (input: InputInfo): string => {
    if (input.type.toLowerCase() === "boolean") {
      return (!!input.value).toString();
    }
    if (input.widget === "combo" && Array.isArray(input.value)) {
      return input.value[0]?.toString() || "";
    }
    return input.value?.toString() || "0";
  };

  // Update the field selection handler
  const handleFieldSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedField = e.target.value;

    const input =
      availableNodes[promptIdxToUpdate][panelState.nodeId]?.inputs[
        selectedField
      ];
    if (input) {
      const initialValue = getInitialValue(input);
      onStateChange({
        fieldName: selectedField,
        value: initialValue,
      });
    } else {
      onStateChange({ fieldName: selectedField });
    }
  };

  // Get controller mapping
  const { getMapping } = useControllerMapping();
  const mapping = getMapping(panelState.nodeId, panelState.fieldName);

  // Modify the sendValueToComfyUI function
  const sendValueToComfyUI = React.useCallback((rawValue: any) => {
    if (!controlChannel || !panelState.isAutoUpdateEnabled || !currentPrompts) return;
    
    console.log("Controller sending value to ComfyUI:", rawValue);
    
    const currentInput = panelState.nodeId && panelState.fieldName 
      ? availableNodes[promptIdxToUpdate][panelState.nodeId]?.inputs[panelState.fieldName] 
      : null;
    
    if (!currentInput) {
      console.warn("Cannot send value - no current input selected");
      return;
    }
    
    // Process the value based on the input type
    let processedValue: InputValue = rawValue;
    
    switch (currentInput.type.toLowerCase()) {
      case "number":
        if (typeof rawValue === 'string') {
          if (!/^-?\d*\.?\d*$/.test(rawValue) || rawValue === "") {
            console.warn("Invalid number value:", rawValue);
            return;
          }
          processedValue = parseFloat(rawValue);
        } else if (typeof rawValue === 'number') {
          processedValue = rawValue;
        } else {
          console.warn("Unexpected value type for number input:", rawValue);
          return;
        }
        
        // Apply min/max constraints
        if (currentInput.min !== undefined && processedValue < currentInput.min) {
          processedValue = currentInput.min;
        }
        if (currentInput.max !== undefined && processedValue > currentInput.max) {
          processedValue = currentInput.max;
        }
        
        break;
        
      case "boolean":
        if (typeof rawValue === 'string') {
          processedValue = rawValue === 'true';
        } else if (typeof rawValue === 'boolean') {
          processedValue = rawValue;
        } else {
          console.warn("Unexpected value type for boolean input:", rawValue);
          return;
        }
        break;
        
      default:
        processedValue = String(rawValue);
    }
    
    console.log("Processed value:", processedValue);
    
    // Update last sent value ref
    lastSentValueRef.current = {
      nodeId: panelState.nodeId,
      fieldName: panelState.fieldName,
      value: processedValue,
    };
    
    // Create a unique key for this update
    const updateKey = `${promptIdxToUpdate}|${panelState.nodeId}|${panelState.fieldName}`;
    
    // Add to the update queue with the processed value
    UPDATE_QUEUE[updateKey] = {
      value: processedValue,
      timestamp: Date.now(),
    };
    
    // Clear any existing timeout
    if (updateTimeoutId) {
      clearTimeout(updateTimeoutId);
    }
    
    // Determine if this is an axis control for more responsive updates
    const isAxisControl = mapping?.type === 'axis';
    const debounceTime = isAxisControl ? AXIS_DEBOUNCE_TIME_MS : DEBOUNCE_TIME_MS;
    
    // Schedule processing of all updates - use shorter or no debounce for axis controls
    updateTimeoutId = setTimeout(() => {
      processUpdateQueue(controlChannel, currentPrompts, setCurrentPrompts);
      updateTimeoutId = null;
    }, debounceTime);
    
  }, [controlChannel, panelState.nodeId, panelState.fieldName, panelState.isAutoUpdateEnabled, 
      availableNodes, promptIdxToUpdate, currentPrompts, setCurrentPrompts, mapping]);

  // Set up controller input handler with enhanced feedback
  useControllerInput(mapping, (value) => {
    console.log("Controller input received:", value);
    
    // For regular values, ALWAYS update the control panel state immediately
    if (typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean') {
      // Get input details to correctly format the value
      const currentInput = panelState.nodeId && panelState.fieldName 
        ? availableNodes[promptIdxToUpdate][panelState.nodeId]?.inputs[panelState.fieldName] 
        : null;
      
      if (!currentInput) {
        console.warn("Cannot format value - no current input selected");
        return;
      }
      
      let formattedValue: string;
      
      // Format the value according to the input type
      switch (currentInput.type.toLowerCase()) {
        case "number":
        case "float":
        case "int": {
          let numValue: number;
          
          if (typeof value === 'number') {
            // If we have a raw number from the controller
            if (mapping?.type === 'axis') {
              const axisMapping = mapping as AxisMapping;
              
              // If min/max are defined in the mapping, use those
              if (axisMapping.minOverride !== undefined && axisMapping.maxOverride !== undefined) {
                // The value should already be scaled in use-controller-input.ts,
                // but let's ensure it respects the bounds
                numValue = Math.min(Math.max(value, axisMapping.minOverride), axisMapping.maxOverride);
              } 
              // Otherwise use the input's min/max if available
              else if (currentInput.min !== undefined && currentInput.max !== undefined) {
                // Map from [-1, 1] to [min, max]
                numValue = currentInput.min + ((value + 1) / 2) * (currentInput.max - currentInput.min);
              } else {
                // If no bounds are defined, just use the raw value
                numValue = value;
              }
            } else {
              numValue = value;
            }
            
            // For integer types, round the value
            if (currentInput.type.toLowerCase() === "int") {
              numValue = Math.round(numValue);
            } else {
              // For float, limit precision to avoid UI jitter
              numValue = parseFloat(numValue.toFixed(4));
            }
          } else {
            // Try to parse string or boolean to number
            numValue = parseFloat(String(value));
            if (isNaN(numValue)) numValue = 0;
          }
          
          // Apply min/max constraints from the input
          if (currentInput.min !== undefined) {
            numValue = Math.max(numValue, currentInput.min);
          }
          if (currentInput.max !== undefined) {
            numValue = Math.min(numValue, currentInput.max);
          }
          
          formattedValue = numValue.toString();
          break;
        }
        
        case "boolean":
          if (typeof value === 'boolean') {
            formattedValue = value.toString();
          } else if (typeof value === 'number') {
            // Treat any non-zero value as true
            formattedValue = (value !== 0).toString();
          } else if (typeof value === 'string') {
            // Use standard JS truthiness for strings
            formattedValue = (value !== '' && value !== '0' && value.toLowerCase() !== 'false').toString();
          } else {
            formattedValue = 'false';
          }
          break;
          
        default:
          // For other types (like string or combo), just convert to string
          formattedValue = String(value);
      }
      
      console.log(`Updating control panel state with formatted value: ${formattedValue} (original: ${value})`);
      
      // Always update local state regardless of auto-update setting
      onStateChange({ value: formattedValue });
      
      // Only send to ComfyUI if auto-update is enabled
      if (panelState.isAutoUpdateEnabled) {
        sendValueToComfyUI(formattedValue);
      } else {
        console.log("Auto-update is disabled, UI updated but not sending to server");
      }
    }
  });

  return (
    <div className="flex flex-col gap-3 p-3">
      <select
        value={promptIdxToUpdate}
        onChange={(e) => setPromptIdxToUpdate(parseInt(e.target.value))}
        className="p-2 border rounded"
      >
        {currentPrompts &&
          currentPrompts.map((_: any, idx: number) => (
            <option key={idx} value={idx}>
              Prompt {idx}
            </option>
          ))}
      </select>
      <select
        value={panelState.nodeId}
        onChange={(e) => {
          onStateChange({
            nodeId: e.target.value,
            fieldName: "",
            value: "0",
          });
        }}
        className="p-2 border rounded"
      >
        <option value="">Select Node</option>
        {Object.entries(availableNodes[promptIdxToUpdate]).map(([id, info]) => (
          <option key={id} value={id}>
            {id} ({info.class_type})
          </option>
        ))}
      </select>

      <select
        value={panelState.fieldName}
        onChange={handleFieldSelect}
        disabled={!panelState.nodeId}
        className="p-2 border rounded"
      >
        <option value="">Select Field</option>
        {panelState.nodeId &&
          availableNodes[promptIdxToUpdate][panelState.nodeId]?.inputs &&
          Object.entries(
            availableNodes[promptIdxToUpdate][panelState.nodeId].inputs,
          )
            .filter(([_, info]) => {
              const type =
                typeof info.type === "string"
                  ? info.type.toLowerCase()
                  : String(info.type).toLowerCase();
              return (
                ["boolean", "number", "float", "int", "string"].includes(
                  type,
                ) || info.widget === "combo"
              );
            })
            .map(([field, info]) => (
              <option key={field} value={field}>
                {field} ({info.type}
                {info.widget ? ` - ${info.widget}` : ""})
              </option>
            ))}
      </select>

      <div className="flex items-center gap-2">
        {panelState.nodeId &&
          panelState.fieldName &&
          availableNodes[promptIdxToUpdate][panelState.nodeId]?.inputs[
            panelState.fieldName
          ] && (
            <InputControl
              input={
                availableNodes[promptIdxToUpdate][panelState.nodeId].inputs[
                  panelState.fieldName
                ]
              }
              value={panelState.value}
              onChange={handleValueChange}
            />
          )}

        {panelState.nodeId &&
          panelState.fieldName &&
          availableNodes[promptIdxToUpdate][panelState.nodeId]?.inputs[
            panelState.fieldName
          ]?.type === "number" && (
            <span className="text-sm text-gray-600">
              {availableNodes[promptIdxToUpdate][panelState.nodeId]?.inputs[
                panelState.fieldName
              ]?.min !== undefined &&
                availableNodes[promptIdxToUpdate][panelState.nodeId]?.inputs[
                  panelState.fieldName
                ]?.max !== undefined &&
                `(${availableNodes[promptIdxToUpdate][panelState.nodeId]?.inputs[panelState.fieldName]?.min} - ${availableNodes[promptIdxToUpdate][panelState.nodeId]?.inputs[panelState.fieldName]?.max})`}
            </span>
          )}

        <ControllerMappingButton
          nodeId={panelState.nodeId}
          fieldName={panelState.fieldName}
          inputType={panelState.nodeId && panelState.fieldName ? availableNodes[promptIdxToUpdate][panelState.nodeId]?.inputs[panelState.fieldName]?.type : ""}
          inputMin={panelState.nodeId && panelState.fieldName ? availableNodes[promptIdxToUpdate][panelState.nodeId]?.inputs[panelState.fieldName]?.min : undefined}
          inputMax={panelState.nodeId && panelState.fieldName ? availableNodes[promptIdxToUpdate][panelState.nodeId]?.inputs[panelState.fieldName]?.max : undefined}
        />
      </div>

      <button
        onClick={toggleAutoUpdate}
        disabled={!controlChannel}
        className={`p-2 rounded ${
          !controlChannel
            ? "bg-gray-300 text-gray-600 cursor-not-allowed"
            : panelState.isAutoUpdateEnabled
              ? "bg-green-500 text-white"
              : "bg-red-500 text-white"
        }`}
      >
        Auto-Update{" "}
        {controlChannel
          ? panelState.isAutoUpdateEnabled
            ? "(ON)"
            : "(OFF)"
          : "(Not Connected)"}
      </button>
    </div>
  );
};
