import { Check, Copy } from 'lucide-react';
import { useMemo, useState } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight, vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';

import { IconButton } from './IconButton';

interface MarkdownContentProps {
  content: string;
  isDark: boolean;
}

export function MarkdownContent({ content, isDark }: MarkdownContentProps) {
  const components = useMemo<Components>(
    () => ({
      code({ className, children, ...props }) {
        const match = /language-(\w+)/.exec(className ?? '');
        const value = String(children).replace(/\n$/, '');

        if (match) {
          return <CodeBlock code={value} language={match[1]} isDark={isDark} />;
        }

        return (
          <code className={className} {...props}>
            {children}
          </code>
        );
      },
    }),
    [isDark],
  );

  return (
    <ReactMarkdown className="markdown" remarkPlugins={[remarkGfm]} components={components}>
      {content}
    </ReactMarkdown>
  );
}

interface CodeBlockProps {
  code: string;
  language: string;
  isDark: boolean;
}

function CodeBlock({ code, language, isDark }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const copyCode = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  };

  return (
    <div className="overflow-hidden rounded-md border border-surface-200 bg-white dark:border-surface-700 dark:bg-surface-900">
      <div className="flex h-10 items-center justify-between border-b border-surface-200 bg-surface-100 px-3 dark:border-surface-700 dark:bg-surface-800">
        <span className="font-mono text-xs text-surface-700 dark:text-surface-200">{language}</span>
        <IconButton label="Copy code" onClick={copyCode} className="h-8 w-8">
          {copied ? <Check size={15} /> : <Copy size={15} />}
        </IconButton>
      </div>
      <SyntaxHighlighter
        language={language}
        style={isDark ? vscDarkPlus : oneLight}
        customStyle={{
          margin: 0,
          padding: '1rem',
          background: 'transparent',
          fontSize: '0.875rem',
          lineHeight: 1.65,
        }}
        codeTagProps={{ style: { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' } }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}
