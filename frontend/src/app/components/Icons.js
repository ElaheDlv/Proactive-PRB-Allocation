import React from 'react';

// Simple SVG icons for repo link and bulb
export const RepoIcon = () => (
  <svg width="16" height="16" fill="currentColor" style={{ display: "inline", verticalAlign: "middle", marginRight: 4 }}>
    <path d="M2 2v12h12V2H2zm1 1h10v10H3V3zm2 2v2h2V5H5zm0 3v2h2V8H5zm3-3v2h2V5H8zm0 3v2h2V8H8z"/>
  </svg>
);

export const BulbIcon = () => (
  <svg width="16" height="16" fill="gold" style={{ display: "inline", verticalAlign: "middle", marginRight: 4 }}>
    <path d="M8 1a5 5 0 0 0-3 9c.01.5.13 1.02.36 1.47.23.45.56.86.97 1.18V14a1 1 0 0 0 2 0v-1.35c.41-.32.74-.73.97-1.18.23-.45.35-.97.36-1.47A5 5 0 0 0 8 1zm0 12a3 3 0 0 1-3-3h6a3 3 0 0 1-3 3z"/>
  </svg>
);
