import React from 'react';
import { motion } from 'framer-motion';
import { Lightbulb, BookOpen, Zap, Star } from 'lucide-react';
import { getSectionTheme } from '../utils/colors';
import './Flashcard.css';

const Flashcard = ({ data, isFlipped, onFlip, colorIndex, style = {}, isFavorite, onToggleFavorite }) => {
    const theme = getSectionTheme(colorIndex);

    const frontStyle = {
        background: theme.background,
        color: 'white',
        border: `1px solid ${theme.borderColor}`,
        boxShadow: theme.shadow
    };

    const backStyle = {
        background: '#0f172a', // Slate 900
        border: '1px solid rgba(255, 255, 255, 0.1)',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)'
    };

    return (
        <motion.div
            className="flashcard-container"
            onClick={onFlip}
            style={style} // Parent controls position/scale/z-index
        >
            <motion.div
                className="flashcard-inner"
                initial={false}
                animate={{
                    rotateY: isFlipped ? 180 : 0,
                    scale: isFlipped ? 1.05 : 1 // Subtle lift during flip status
                }}
                transition={{
                    duration: 0.8,
                    type: "spring",
                    stiffness: 180,
                    damping: 15,
                    mass: 0.8
                }}
                style={{ transformStyle: 'preserve-3d' }}
            >
                {/* Front */}
                <div className="flashcard-face flashcard-front" style={frontStyle}>
                    <div className="card-header">
                        <span className="learning-badge" style={{ color: theme.accentColor, background: 'rgba(255,255,255,0.1)' }}>
                            Learning
                        </span>

                        {onToggleFavorite && (
                            <button
                                className="star-btn"
                                onClick={onToggleFavorite}
                                onMouseDown={(e) => e.stopPropagation()} // Prevent flip
                                aria-label={isFavorite ? "Unfavorite" : "Favorite"}
                            >
                                <Star
                                    size={22}
                                    fill={isFavorite ? "#fbbf24" : "none"}
                                    color={isFavorite ? "#fbbf24" : "rgba(255,255,255,0.6)"}
                                    strokeWidth={isFavorite ? 0 : 2}
                                />
                            </button>
                        )}
                    </div>

                    <div className="card-content-front">
                        <h2>{data.question}</h2>
                        <div className="icon-decoration">
                            <Zap size={48} strokeWidth={1} opacity={0.3} />
                        </div>
                    </div>

                    <div className="card-footer-action">
                        <button className="tap-btn">Tap to flip</button>
                    </div>
                </div>

                {/* Back */}
                <div className="flashcard-face flashcard-back" style={backStyle}>
                    <div className="card-content-back">
                        <div className="section definition">
                            <h3><BookOpen size={18} /> Definition</h3>
                            <p>{data.definition}</p>
                        </div>

                        {data.example && (
                            <div className="section example">
                                <h3><Zap size={18} /> Example</h3>
                                <p>{data.example}</p>
                            </div>
                        )}

                        {data.tip && (
                            <div className="section tip">
                                <h3><Lightbulb size={18} /> Quick Tip</h3>
                                <p>{data.tip}</p>
                            </div>
                        )}
                    </div>
                </div>
            </motion.div>
        </motion.div>
    );
};

export default Flashcard;
