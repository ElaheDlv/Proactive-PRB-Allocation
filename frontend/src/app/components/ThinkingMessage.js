import React from 'react';

/**
 * ThinkingMessage component: Animated dots to indicate the assistant is thinking.
 */
function ThinkingMessage() {
  return (
    <div className="flex items-center gap-2">
      <span className="font-semibold text-gray-400">Assistant is thinking</span>
      <span className="dot-flashing" style={{
        display: "inline-block",
        width: 24,
        height: 12,
        position: "relative"
      }}>
        {/* Individual dots for animation */}
        <span style={{
          position: "absolute",
          left: 0,
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: "#888",
          animation: "dotFlashing 1s infinite linear alternate"
        }} />
        <span style={{
          position: "absolute",
          left: 9,
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: "#888",
          animation: "dotFlashing 1s infinite linear alternate 0.3s"
        }} />
        <span style={{
          position: "absolute",
          left: 18,
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: "#888",
          animation: "dotFlashing 1s infinite linear alternate 0.6s"
        }} />
        {/* CSS for the dot flashing animation */}
        <style jsx>{`
          @keyframes dotFlashing {
            0% { opacity: 0.2; }
            50%, 100% { opacity: 1; }
          }
        `}</style>
      </span>
    </div>
  );
}

export default ThinkingMessage;
