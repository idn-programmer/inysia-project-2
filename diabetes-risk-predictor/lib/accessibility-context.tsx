"use client"

import React, { createContext, useContext, useEffect, useState } from 'react'

interface AccessibilityPreferences {
  zoomLevel: number
  highContrast: boolean
}

interface AccessibilityContextType {
  zoomLevel: number
  highContrast: boolean
  setZoomLevel: (level: number) => void
  setHighContrast: (enabled: boolean) => void
  cycleZoomLevel: () => void
  toggleHighContrast: () => void
}

const AccessibilityContext = createContext<AccessibilityContextType | undefined>(undefined)

const STORAGE_KEY = 'accessibility-preferences'
const ZOOM_LEVELS = [100, 125, 150, 175] as const

export function AccessibilityProvider({ children }: { children: React.ReactNode }) {
  const [preferences, setPreferences] = useState<AccessibilityPreferences>({
    zoomLevel: 100,
    highContrast: false,
  })

  // Load preferences from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) {
        const parsed = JSON.parse(stored) as AccessibilityPreferences
        setPreferences(parsed)
      }
    } catch (error) {
      console.warn('Failed to load accessibility preferences:', error)
    }
  }, [])

  // Save preferences to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences))
    } catch (error) {
      console.warn('Failed to save accessibility preferences:', error)
    }
  }, [preferences])

  const setZoomLevel = (level: number) => {
    setPreferences(prev => ({ ...prev, zoomLevel: level }))
  }

  const setHighContrast = (enabled: boolean) => {
    setPreferences(prev => ({ ...prev, highContrast: enabled }))
  }

  const cycleZoomLevel = () => {
    const currentIndex = ZOOM_LEVELS.indexOf(preferences.zoomLevel as typeof ZOOM_LEVELS[number])
    const nextIndex = (currentIndex + 1) % ZOOM_LEVELS.length
    setZoomLevel(ZOOM_LEVELS[nextIndex])
  }

  const toggleHighContrast = () => {
    setHighContrast(!preferences.highContrast)
  }

  const value: AccessibilityContextType = {
    zoomLevel: preferences.zoomLevel,
    highContrast: preferences.highContrast,
    setZoomLevel,
    setHighContrast,
    cycleZoomLevel,
    toggleHighContrast,
  }

  return (
    <AccessibilityContext.Provider value={value}>
      {children}
    </AccessibilityContext.Provider>
  )
}

export function useAccessibility() {
  const context = useContext(AccessibilityContext)
  if (context === undefined) {
    throw new Error('useAccessibility must be used within an AccessibilityProvider')
  }
  return context
}
