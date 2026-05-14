import * as React from "react";

import { cn } from "@/lib/utils";

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        "flex h-8 w-full min-w-0 rounded-md border border-gray-600 bg-gray-900 text-xs text-gray-100 shadow-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 placeholder:text-gray-500 disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      {...props}
    />
  );
}

export { Input };
