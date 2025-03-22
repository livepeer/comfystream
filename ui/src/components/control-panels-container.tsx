"use client";

import React, { useState, useEffect } from "react";
import { ControlPanel } from "./control-panel";
import { Button } from "./ui/button";
import { Drawer, DrawerContent, DrawerTitle } from "./ui/drawer";
import { Settings } from "lucide-react";
import { Plus } from "lucide-react"; // Import Plus icon for minimal add button

// Custom event name for control panel refresh
const CONTROL_PANEL_REFRESH_EVENT = 'comfystream:refreshControlPanel';

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
            {/* Left side add button */}
            <div className="w-12 border-r flex items-start pt-4 justify-center bg-background/50">
              <Button
                onClick={addPanel}
                variant="ghost"
                size="icon"
                className="h-8 w-8 rounded-md bg-blue-500 hover:bg-blue-600 active:bg-blue-700 transition-colors shadow-sm hover:shadow text-white"
                title="Add control panel"
                aria-label="Add control panel"
                data-tooltip="Add control panel"
              >
                <Plus className="h-4 w-4" />
              </Button>
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
