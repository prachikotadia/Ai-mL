export const gradients = [
    { name: 'Sunset', from: '#ff9a9e', to: '#fecfef', darkFrom: '#4c1d23', darkTo: '#2a1b2e' },
    { name: 'Ocean', from: '#a18cd1', to: '#fbc2eb', darkFrom: '#241b4c', darkTo: '#3a2b45' },
    { name: 'Forest', from: '#84fab0', to: '#8fd3f4', darkFrom: '#0f3a2b', darkTo: '#1b3a45' },
    { name: 'Nebula', from: '#cfd9df', to: '#e2ebf0', darkFrom: '#2e3346', darkTo: '#1b1e2b' }, // Subtle
    { name: 'Midnight', from: '#667eea', to: '#764ba2', darkFrom: '#1e2156', darkTo: '#2d1b4e' },
    { name: 'Royal', from: '#b721ff', to: '#21d4fd', darkFrom: '#3a0c56', darkTo: '#0c4a56' },
    { name: 'Gold', from: '#fa709a', to: '#fee140', darkFrom: '#562b3a', darkTo: '#564a1b' },
    { name: 'Aurora', from: '#43e97b', to: '#38f9d7', darkFrom: '#174e2d', darkTo: '#134e45' },
];

export const getSectionTheme = (index) => {
    const theme = gradients[index % gradients.length];

    // Premium Glassmorphism Theme
    return {
        background: `linear-gradient(135deg, ${theme.darkFrom}, ${theme.darkTo})`,
        accentColor: theme.from, // Use lighter color for accents/text
        borderColor: `rgba(255, 255, 255, 0.1)`,
        shadow: `0 8px 32px 0 rgba(0, 0, 0, 0.3)`
    };
};

export const getGlassStyle = (index) => {
    const theme = getSectionTheme(index);
    return {
        background: theme.background,
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        border: `1px solid ${theme.borderColor}`,
        boxShadow: theme.shadow,
        color: 'white'
    };
};
