module.exports = {
  textbookSidebar: [
    'intro',
    {
      type: 'category',
      label: '第一部分：基础入门篇',
      items: [
        'part1-foundations/ch01-overview',
        'part1-foundations/ch02-embodied-intro',
        'part1-foundations/ch03-math-foundations',
        'part1-foundations/ch04-dev-env',
      ],
    },
    {
      type: 'category',
      label: '第二部分：运动系统篇',
      items: [
        'part2-motion/ch05-hardware',
        'part2-motion/ch06-kinematics',
        'part2-motion/ch07-dynamics',
        'part2-motion/ch08-walking',
      ],
    },
    {
      type: 'category',
      label: '第三部分：感知与交互篇',
      items: [
        'part3-perception/ch09-vision',
        'part3-perception/ch10-multimodal',
        'part3-perception/ch11-scene',
        'part3-perception/ch12-hri',
      ],
    },
    {
      type: 'category',
      label: '第四部分：具身智能核心篇',
      items: [
        'part4-core-ei/ch13-rl',
        'part4-core-ei/ch14-imitation',
        'part4-core-ei/ch15-sim2real',
        'part4-core-ei/ch16-world-model',
        'part4-core-ei/ch17-vla',
      ],
    },
    {
      type: 'category',
      label: '第五部分：系统集成与实操项目篇',
      items: [
        'part5-integration/ch18-ros2',
        'part5-integration/ch19-navigation',
        'part5-integration/ch20-collaboration',
        'part5-integration/ch21-deployment',
      ],
    },
    {
      type: 'category',
      label: '第六部分：前沿拓展篇',
      items: [
        'part6-frontiers/ch22-genai',
        'part6-frontiers/ch23-frontiers',
        'part6-frontiers/ch24-safety',
        'part6-frontiers/ch25-general',
      ],
    },
    {
      type: 'category',
      label: '附录',
      items: ['appendices/index'],
    },
  ],
};
