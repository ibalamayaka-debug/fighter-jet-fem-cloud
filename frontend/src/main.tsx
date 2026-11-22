import React from 'react';
import ReactDOM from 'react-dom/client';
import { App } from './App';
import './index.css';

class RootErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 28, color: '#fdf6ee', background: '#16110d', minHeight: '100vh' }}>
          <h2>WebGL Viewer Error</h2>
          <pre style={{ whiteSpace: 'pre-wrap', color: '#fb7185' }}>{this.state.error.message}</pre>
        </div>
      );
    }
    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <RootErrorBoundary>
    <App />
  </RootErrorBoundary>
);
