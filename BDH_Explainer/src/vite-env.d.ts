/// <reference types="svelte" />
/// <reference types="vite/client" />

declare module '*.svelte' {
  import type { ComponentType } from 'svelte';
  const component: ComponentType;
  export default component;
}

declare module 'katex' {
  const katex: {
    render(tex: string, element: HTMLElement, options?: any): void;
    renderToString(tex: string, options?: any): string;
  };
  export default katex;
}
