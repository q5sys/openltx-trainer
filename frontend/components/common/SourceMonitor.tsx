/**
 * SourceMonitor: reusable video/image preview component.
 *
 * Provides play/pause, seek, frame-by-frame stepping, and current time display.
 * Used in the Dataset Cutter for reviewing source media before clipping.
 */

import { useState, useRef, useCallback, useEffect } from 'react'
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react'
import { Button } from '../ui/button'

interface SourceMonitorProps {
  /** URL or local file path to the media (video or image). */
  src: string
  /** Whether the source is an image (no playback controls). */
  isImage?: boolean
  /** Called when the user clicks to set an in-point (seconds). */
  onSetInPoint?: (time: number) => void
  /** Called when the user clicks to set an out-point (seconds). */
  onSetOutPoint?: (time: number) => void
  /** Optional CSS class for the container. */
  className?: string
}

export function SourceMonitor({
  src,
  isImage = false,
  onSetInPoint,
  onSetOutPoint,
  className = '',
}: SourceMonitorProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)

  const togglePlay = useCallback(() => {
    const video = videoRef.current
    if (!video) return

    if (video.paused) {
      video.play()
      setIsPlaying(true)
    } else {
      video.pause()
      setIsPlaying(false)
    }
  }, [])

  const stepFrame = useCallback((direction: number) => {
    const video = videoRef.current
    if (!video) return
    // Assume ~24fps, step by one frame duration.
    const frameDuration = 1 / 24
    video.currentTime = Math.max(0, Math.min(video.duration, video.currentTime + direction * frameDuration))
  }, [])

  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current
    if (video) {
      setCurrentTime(video.currentTime)
    }
  }, [])

  const handleLoadedMetadata = useCallback(() => {
    const video = videoRef.current
    if (video) {
      setDuration(video.duration)
    }
  }, [])

  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current
    if (!video) return
    const time = parseFloat(e.target.value)
    video.currentTime = time
    setCurrentTime(time)
  }, [])

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    const frames = Math.floor((seconds % 1) * 24)
    return `${mins}:${secs.toString().padStart(2, '0')}:${frames.toString().padStart(2, '0')}`
  }

  useEffect(() => {
    setIsPlaying(false)
    setCurrentTime(0)
    setDuration(0)
  }, [src])

  if (isImage) {
    return (
      <div className={`flex flex-col items-center gap-2 ${className}`}>
        <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
          <img
            src={src}
            alt="Source preview"
            className="w-full h-full object-contain"
          />
        </div>
        <p className="text-xs text-muted-foreground">Still image (no playback)</p>
      </div>
    )
  }

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      {/* Video viewport */}
      <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
        <video
          ref={videoRef}
          src={src}
          className="w-full h-full object-contain"
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={() => setIsPlaying(false)}
          preload="metadata"
        />
      </div>

      {/* Seek bar */}
      <input
        type="range"
        min={0}
        max={duration || 0}
        step={0.001}
        value={currentTime}
        onChange={handleSeek}
        className="w-full h-1.5 accent-primary cursor-pointer"
      />

      {/* Transport controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="sm" onClick={() => stepFrame(-1)} title="Previous frame">
            <SkipBack className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="sm" onClick={togglePlay} title={isPlaying ? 'Pause' : 'Play'}>
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </Button>
          <Button variant="ghost" size="sm" onClick={() => stepFrame(1)} title="Next frame">
            <SkipForward className="h-4 w-4" />
          </Button>
        </div>

        <span className="text-xs font-mono text-muted-foreground">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>

        <div className="flex items-center gap-1">
          {onSetInPoint && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => onSetInPoint(currentTime)}
              title="Set in-point at current time"
            >
              In
            </Button>
          )}
          {onSetOutPoint && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => onSetOutPoint(currentTime)}
              title="Set out-point at current time"
            >
              Out
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
