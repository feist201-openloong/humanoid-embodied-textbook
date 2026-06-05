import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: '人形机器人与具身智能：从入门到精通',
  tagline: 'Humanoid Robots & Embodied Intelligence: From Foundations to Frontiers',
  favicon: 'img/favicon.ico',
  url: 'https://your-username.github.io',
  baseUrl: '/humanoid-embodied-textbook/',
  organizationName: 'your-username',
  projectName: 'humanoid-embodied-textbook',
  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',
  i18n: {
    defaultLocale: 'zh-CN',
    locales: ['zh-CN', 'en'],
  },
  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/your-username/humanoid-embodied-textbook/edit/main/website/',
          showLastUpdateTime: true,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],
  themeConfig: {
    image: 'img/og-image.png',
    navbar: {
      title: '人形机器人与具身智能',
      logo: {
        alt: 'Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'textbookSidebar',
          position: 'left',
          label: '目录',
        },
        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          href: 'https://github.com/your-username/humanoid-embodied-textbook',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: '教材',
          items: [
            {label: '目录', to: '/docs/intro'},
            {label: 'GitHub 仓库', href: 'https://github.com/your-username/humanoid-embodied-textbook'},
          ],
        },
        {
          title: '更多',
          items: [
            {label: 'GitHub', href: 'https://github.com/your-username/humanoid-embodied-textbook'},
          ],
        },
      ],
      copyright: `Copyright © 2025 本书采用 CC BY-NC-SA 4.0 协议开源`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'yaml', 'xml'],
    },
    mermaid: {
      theme: {light: 'neutral', dark: 'dark'},
    },
  } satisfies Preset.ThemeConfig,
  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],
};

export default config;
