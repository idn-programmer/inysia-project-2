"use client"

import { ZoomIn, Contrast } from "lucide-react"
import { useAccessibility } from "@/lib/accessibility-context"
import { cn } from "@/lib/utils"

export function AccessibilityControls() {
  const { zoomLevel, highContrast, cycleZoomLevel, toggleHighContrast } = useAccessibility()

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
      {/* Zoom Button */}
      <button
        onClick={cycleZoomLevel}
        className={cn(
          "flex items-center gap-2 px-3 py-2 rounded-lg shadow-lg",
          "bg-background border border-border hover:bg-muted",
          "transition-all duration-200 hover:shadow-xl",
          "text-sm font-medium"
        )}
        aria-label={`Current zoom: ${zoomLevel}%. Click to change zoom level`}
        title={`Current zoom: ${zoomLevel}%. Click to cycle through zoom levels`}
      >
        <ZoomIn className="size-4" aria-hidden="true" />
        <span className="hidden sm:inline">{zoomLevel}%</span>
      </button>

      {/* High Contrast Button */}
      <button
        onClick={toggleHighContrast}
        className={cn(
          "flex items-center gap-2 px-3 py-2 rounded-lg shadow-lg",
          "bg-background border border-border hover:bg-muted",
          "transition-all duration-200 hover:shadow-xl",
          "text-sm font-medium",
          highContrast && "bg-primary text-primary-foreground border-primary"
        )}
        aria-label={highContrast ? "Disable high contrast mode" : "Enable high contrast mode"}
        title={highContrast ? "High contrast mode is ON. Click to disable." : "High contrast mode is OFF. Click to enable."}
      >
        <Contrast className="size-4" aria-hidden="true" />
        <span className="hidden sm:inline">
          {highContrast ? "High Contrast" : "Normal"}
        </span>
      </button>
    </div>
  )
}
