import { motion } from 'framer-motion';

const dotTransition = {
  repeat: Infinity,
  duration: 0.9,
  ease: 'easeInOut',
} as const;

export function TypingIndicator() {
  return (
    <div className="flex items-center gap-2 text-sm text-surface-700 dark:text-surface-200">
      <span>GenPy is thinking</span>
      <span className="flex gap-1">
        {[0, 1, 2].map((index) => (
          <motion.span
            key={index}
            className="h-1.5 w-1.5 rounded-full bg-brand-500"
            animate={{ y: [0, -4, 0], opacity: [0.45, 1, 0.45] }}
            transition={{ ...dotTransition, delay: index * 0.14 }}
          />
        ))}
      </span>
    </div>
  );
}
