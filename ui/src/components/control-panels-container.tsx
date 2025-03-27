"use client";

import React, { useState, useEffect } from "react";
import { ControlPanel } from "./control-panel";
import { Button } from "./ui/button";
import { Drawer, DrawerContent, DrawerTitle } from "./ui/drawer";
import { Settings, Save, Upload, Plus } from "lucide-react"; // Import all icons
import { useControllerContext } from "@/context/controller-context";
import { ControllerMapping } from "@/types/controller";

// Define the unified configuration type
interface ComfyStreamConfig {
  version: string;
  panels: Array<{
    nodeId: string;
    fieldName: string;
    value: string;
    isAutoUpdateEnabled: boolean;
  }>;
  mappings: Record<string, Record<string, ControllerMapping>>;
}

// Custom event name for control panel refresh
const CONTROL_PANEL_REFRESH_EVENT = 'comfystream:refreshControlPanel';
// Custom event for loading controller mappings
const LOAD_MAPPINGS_EVENT = 'comfystream:loadMappings';

export const ControlPanelsContainer = () => {
  const [panels, setPanels] = useState<number[]>([0]); // Start with one panel
  const [nextPanelId, setNextPanelId] = useState(1);
  const [isOpen, setIsOpen] = useState(false);
  const [forceUpdateKey, setForceUpdateKey] = useState(0); // Add a key for forcing re-renders
  const [panelStates, setPanelStates] = useState<
    Record<
      number,
      {
        nodeId: string;
        fieldName: string;
        value: string;
        isAutoUpdateEnabled: boolean;
      }
    >
  >({
    0: {
      nodeId: "",
      fieldName: "",
      value: "0",
      isAutoUpdateEnabled: false,
    },
  });

  // Get the controller mappings from context
  const { mappings } = useControllerContext();

  // Listen for refresh event to force control panel to close and reopen
  useEffect(() => {
    const handleRefresh = () => {
      console.log('Control panel refresh event received');
      
      // First, increment the force update key to trigger a complete re-render
      setForceUpdateKey(prev => prev + 1);
      
      // Then close and reopen the panel with a small delay
      setIsOpen(false);
      
      setTimeout(() => {
        setIsOpen(true);
      }, 100);
    };

    // Add event listener
    window.addEventListener(CONTROL_PANEL_REFRESH_EVENT, handleRefresh);

    // Cleanup
    return () => {
      window.removeEventListener(CONTROL_PANEL_REFRESH_EVENT, handleRefresh);
    };
  }, []);

  const addPanel = () => {
    const newId = nextPanelId;
    setPanels([...panels, newId]);
    setPanelStates((prev) => ({
      ...prev,
      [newId]: {
        nodeId: "",
        fieldName: "",
        value: "0",
        isAutoUpdateEnabled: false,
      },
    }));
    setNextPanelId(nextPanelId + 1);
  };

  const removePanel = (id: number) => {
    setPanels(panels.filter((panelId) => panelId !== id));
    setPanelStates((prev) => {
      const newState = { ...prev };
      delete newState[id];
      return newState;
    });
  };

  const updatePanelState = (
    id: number,
    state: Partial<(typeof panelStates)[number]>,
  ) => {
    setPanelStates((prev) => ({
      ...prev,
      [id]: {
        ...prev[id],
        ...state,
      },
    }));
  };

  // Export panels configuration and controller mappings to a unified JSON file
  const exportConfig = () => {
    try {
      // Create unified configuration object
      const config: ComfyStreamConfig = {
        version: "1.0.0",
        panels: Object.values(panelStates).map(state => ({
          nodeId: state.nodeId,
          fieldName: state.fieldName,
          value: state.value,
          isAutoUpdateEnabled: state.isAutoUpdateEnabled
        })),
        mappings: mappings // Include controller mappings from context
      };
      
      // Convert to JSON and create download
      const jsonString = JSON.stringify(config, null, 2);
      const blob = new Blob([jsonString], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      
      // Create download link
      const a = document.createElement("a");
      a.href = url;
      a.download = "comfystream-config.json";
      document.body.appendChild(a);
      a.click();
      
      // Clean up
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      console.log("ComfyStream configuration exported successfully");
    } catch (error) {
      console.error("Failed to export configuration:", error);
      alert("Failed to save configuration. Please try again.");
    }
  };

  // Import unified configuration from a JSON file
  const importConfig = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const config = JSON.parse(e.target?.result as string) as ComfyStreamConfig;
        
        // Check version compatibility
        if (!config.version) {
          alert("Invalid configuration file format");
          return;
        }
        
        // Create new panel states
        const newPanelStates: typeof panelStates = {};
        const newPanels: number[] = [];
        
        if (config.panels) {
          config.panels.forEach((panel, index) => {
            newPanelStates[index] = {
              nodeId: panel.nodeId,
              fieldName: panel.fieldName,
              value: panel.value,
              isAutoUpdateEnabled: panel.isAutoUpdateEnabled
            };
            newPanels.push(index);
          });
          
          // Update panel state
          setPanels(newPanels);
          setPanelStates(newPanelStates);
          setNextPanelId(config.panels.length);
        }
        
        // Load controller mappings if present
        if (config.mappings) {
          // Dispatch event to update controller mappings
          // This event will be caught by the useControllerMapping hook
          window.dispatchEvent(new CustomEvent(LOAD_MAPPINGS_EVENT, {
            detail: { mappings: config.mappings }
          }));
          
          console.log("Controller mappings loaded from configuration file");
        }
        
        // Force refresh
        setForceUpdateKey(prev => prev + 1);
        
        console.log("ComfyStream configuration imported successfully");
      } catch (error) {
        console.error("Failed to parse configuration file:", error);
        alert("Failed to load configuration. The file may be corrupted or in an invalid format.");
      }
    };
    
    reader.readAsText(file);
    
    // Reset the input so the same file can be selected again
    event.target.value = '';
  };

  return (
    <>
      <Button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-4 right-4 h-12 w-12 rounded-full p-0 shadow-lg hover:shadow-xl transition-shadow"
        variant="default"
      >
        <Settings className="h-6 w-6" />
      </Button>

      {/* Hidden container with all control panels to keep them active when drawer is closed */}
      {!isOpen && (
        <div 
          key={`hidden-panels-${forceUpdateKey}`}
          aria-hidden="true"
          className="absolute opacity-0 pointer-events-none invisible h-0 overflow-hidden"
          style={{ position: "absolute", left: "-9999px" }}
        >
          {panels.map((id) => (
            <ControlPanel
              key={`hidden-panel-${id}-${forceUpdateKey}`}
              panelState={panelStates[id]}
              onStateChange={(state) => updatePanelState(id, state)}
            />
          ))}
        </div>
      )}

      <Drawer
        key={`drawer-${forceUpdateKey}`}
        open={isOpen}
        onOpenChange={setIsOpen}
        direction="bottom"
        shouldScaleBackground={false}
      >
        {/* This is a hack to remove the background color of the overlay so the screen is not dimmed when the drawer is open */}
        <style>
          {`
            [data-vaul-overlay] {
              background-color: transparent !important;
              background: transparent !important;
            }
            
            /* Custom scrollbar styling */
            #control-panel-drawer ::-webkit-scrollbar {
              width: 8px;
              height: 8px;
            }
            
            #control-panel-drawer ::-webkit-scrollbar-track {
              background: transparent;
            }
            
            #control-panel-drawer ::-webkit-scrollbar-thumb {
              background: #cbd5e1;
              border-radius: 4px;
            }
            
            #control-panel-drawer ::-webkit-scrollbar-thumb:hover {
              background: #94a3b8;
            }
          `}
        </style>
        <DrawerContent
          id="control-panel-drawer"
          className="max-h-[50vh] min-h-[200px] bg-background/90 backdrop-blur-md border-t shadow-lg overflow-hidden"
        >
          <DrawerTitle className="sr-only">Control Panels</DrawerTitle>

          <div className="flex h-full">
            {/* Left side buttons column */}
            <div className="w-12 border-r flex flex-col items-center pt-4 gap-3 bg-background/50">
              {/* Add panel button */}
              <Button
                onClick={addPanel}
                variant="ghost"
                size="icon"
                className="h-8 w-8 rounded-md bg-blue-500 hover:bg-blue-600 active:bg-blue-700 transition-colors shadow-sm hover:shadow text-white"
                title="Add control panel"
                aria-label="Add control panel"
              >
                <Plus className="h-4 w-4" />
              </Button>
              
              {/* Export configuration button */}
              <Button
                onClick={exportConfig}
                variant="ghost"
                size="icon"
                className="h-8 w-8 rounded-md bg-green-500 hover:bg-green-600 active:bg-green-700 transition-colors shadow-sm hover:shadow text-white"
                title="Save complete configuration (panels and mappings)"
                aria-label="Save configuration"
              >
                <Save className="h-4 w-4" />
              </Button>
              
              {/* Import configuration button */}
              <label htmlFor="import-config" className="cursor-pointer">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 rounded-md bg-purple-500 hover:bg-purple-600 active:bg-purple-700 transition-colors shadow-sm hover:shadow text-white"
                  title="Load complete configuration (panels and mappings)"
                  aria-label="Load configuration"
                  asChild
                >
                  <span>
                    <Upload className="h-4 w-4" />
                  </span>
                </Button>
              </label>
              <input
                id="import-config"
                type="file"
                accept=".json"
                onChange={importConfig}
                className="hidden"
              />
            </div>

            {/* Control panels container */}
            <div className="flex-1 overflow-x-auto">
              <div className="flex gap-4 p-4 min-h-0">
                {panels.map((id) => (
                  <div
                    key={`panel-container-${id}-${forceUpdateKey}`}
                    className="flex-none w-80 border rounded-lg bg-white/95 shadow-sm hover:shadow-md transition-shadow overflow-hidden flex flex-col max-h-[calc(50vh-3rem)]"
                  >
                    <div className="flex justify-between items-center p-2 border-b bg-gray-50/80">
                      <span className="font-medium">
                        Control Panel {id + 1}
                      </span>
                      <Button
                        onClick={() => removePanel(id)}
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 rounded-full p-0 hover:bg-gray-200/80 transition-colors"
                      >
                        <span className="text-sm">×</span>
                      </Button>
                    </div>
                    <div className="flex-1 overflow-y-auto">
                      <ControlPanel
                        key={`visible-panel-${id}-${forceUpdateKey}`}
                        panelState={panelStates[id]}
                        onStateChange={(state) => updatePanelState(id, state)}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </DrawerContent>
      </Drawer>
    </>
  );
};
