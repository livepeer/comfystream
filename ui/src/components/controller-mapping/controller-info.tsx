"use client";

import React from 'react';
import { Button } from '@/components/ui/button';
import { useController } from '@/hooks/use-controller';

export function ControllerInfo() {
  const { controllers, isControllerConnected } = useController();
  
  // Debug button to manually check controllers
  const checkControllers = () => {
    console.log("Manual check for controllers:");
    console.log(`Controllers detected: ${controllers.length}`);
    controllers.forEach((c, i) => {
      console.log(`Controller ${i}: ${c.id} - ${c.buttons.length} buttons, ${c.axes.length} axes`);
    });
    
    // Attempt to force refresh controllers
    if (typeof navigator.getGamepads === 'function') {
      const gamepads = navigator.getGamepads();
      console.log(`Raw gamepad API returned ${gamepads.length} gamepads`);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        {isControllerConnected ? (
          <span className="text-xs text-green-500">Controller Connected</span>
        ) : (
          <span className="text-xs text-red-500">No Controller Detected</span>
        )}
        <Button 
          variant="outline" 
          size="sm" 
          onClick={checkControllers}
        >
          Refresh Controllers
        </Button>
      </div>
      
      {/* Debug Info */}
      <div className="bg-gray-100 p-2 rounded text-xs">
        <p>Controllers: {controllers.length}</p>
        {controllers.map((controller, idx) => (
          <div key={idx} className="mt-1">
            <p>#{idx}: {controller.id}</p>
            <p>{controller.buttons.length} buttons, {controller.axes.length} axes</p>
          </div>
        ))}
      </div>
    </div>
  );
} 