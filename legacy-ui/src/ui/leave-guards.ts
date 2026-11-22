/** Shared by app navigation, browser history, logout and document unload. */
export class LeaveGuards {
  private guards = new Set<() => string | null>();
  register(guard: () => string | null): () => void {
    this.guards.add(guard);
    return () => { this.guards.delete(guard); };
  }
  message(): string | null {
    for (const guard of this.guards) {
      const message = guard();
      if (message) return message;
    }
    return null;
  }
  confirm(ask: (message: string) => boolean): boolean {
    const message = this.message();
    return message === null || ask(message);
  }
}
