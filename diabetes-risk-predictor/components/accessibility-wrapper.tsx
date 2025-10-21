"use client"

import { AccessibilityControls } from "@/components/accessibility-controls"
import { useAccessibility } from "@/lib/accessibility-context"
import { useEffect } from "react"

export function AccessibilityWrapper() {
  const { zoomLevel, highContrast } = useAccessibility()
  
  useEffect(() => {
    // Apply zoom level to html element
    document.documentElement.style.fontSize = `${zoomLevel}%`
    
    // Apply high contrast class to html element
    if (highContrast) {
      document.documentElement.classList.add('high-contrast')
    } else {
      document.documentElement.classList.remove('high-contrast')
    }
    
    // Cleanup function to reset styles when component unmounts
    return () => {
      document.documentElement.style.fontSize = ''
      document.documentElement.classList.remove('high-contrast')
    }
  }, [zoomLevel, highContrast])
  
  return <AccessibilityControls />
}

