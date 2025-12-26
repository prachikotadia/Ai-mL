import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { BookOpen, CheckCircle, ArrowRight, Star } from 'lucide-react';
import { motion } from 'framer-motion';
import cardsData from '../data/cards.json';
import { getProgress } from '../utils/storage';
import { getSectionTheme, getGlassStyle } from '../utils/colors';
import './Home.css';

const Home = () => {
    const [progress, setProgress] = useState({});

    useEffect(() => {
        setProgress(getProgress());
    }, []);

    return (
        <div className="home-container">
            <div className="ambient-background">
                <div className="orb orb-1"></div>
                <div className="orb orb-2"></div>
            </div>

            <header className="home-header">
                <h1>AI Interview Flashcards</h1>
                <p>Master ML, LLM & AI Concepts</p>
            </header>

            <motion.div
                className="section-list"
                initial="hidden"
                animate="visible"
                variants={{
                    visible: { transition: { staggerChildren: 0.05 } }
                }}
            >
                {/* Favorites Card */}
                <motion.div
                    variants={{
                        hidden: { opacity: 0, y: 20 },
                        visible: { opacity: 1, y: 0 }
                    }}
                >
                    <Link
                        to="/favorites"
                        className="section-card glass-card"
                        style={{
                            background: 'linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(245, 158, 11, 0.1))',
                            border: '1px solid rgba(251, 191, 36, 0.3)',
                            boxShadow: '0 8px 32px 0 rgba(251, 191, 36, 0.1)',
                            marginBottom: '24px' // Add spacing below
                        }}
                    >
                        <div className="section-info">
                            <h2 style={{ color: '#fbbf24', textShadow: '0 2px 4px rgba(0,0,0,0.2)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <Star size={24} fill="#fbbf24" strokeWidth={0} /> Favorites
                            </h2>
                            <div className="section-meta">
                                <span className="card-count" style={{ color: 'rgba(255,255,255,0.7)' }}>Review your starred cards</span>
                            </div>
                        </div>
                        <div className="card-arrow">
                            <ArrowRight size={20} color="#fbbf24" />
                        </div>
                    </Link>
                </motion.div>

                {cardsData.map((section, index) => {
                    const sectionProgress = progress[section.id] || { knownCardIds: [] };
                    const knownCount = sectionProgress.knownCardIds ? sectionProgress.knownCardIds.length : 0;
                    const totalCards = section.itemCount;
                    const percent = Math.round((knownCount / totalCards) * 100);

                    let status = "Not Started";
                    if (percent === 100) status = "Completed";
                    else if (percent > 0) status = "In Progress";

                    const theme = getSectionTheme(index);
                    const glassStyle = getGlassStyle(index);

                    return (
                        <motion.div
                            key={section.id}
                            variants={{
                                hidden: { opacity: 0, y: 20 },
                                visible: { opacity: 1, y: 0 }
                            }}
                        >
                            <Link
                                to={`/section/${section.id}`}
                                className="section-card glass-card"
                                style={glassStyle}
                            >
                                <div className="section-info">
                                    <h2 style={{ color: 'white', textShadow: '0 2px 4px rgba(0,0,0,0.2)' }}>{section.title}</h2>
                                    <div className="section-meta">
                                        <span className="card-count" style={{ color: 'rgba(255,255,255,0.7)' }}>{totalCards} cards</span>
                                        <span className={`status-badge ${status.toLowerCase().replace(' ', '-')}`}>
                                            {status}
                                        </span>
                                    </div>
                                </div>

                                <div className="progress-container">
                                    <div className="progress-labels">
                                        <span style={{ color: 'rgba(255,255,255,0.8)' }}>{knownCount} / {totalCards} learned</span>
                                        <span style={{ color: 'white', fontWeight: 'bold' }}>{percent}%</span>
                                    </div>
                                    <div className="progress-bar-bg" style={{ background: 'rgba(255,255,255,0.2)' }}>
                                        <div
                                            className="progress-bar-fill"
                                            style={{
                                                width: `${percent}%`,
                                                background: 'white',
                                                boxShadow: '0 0 10px rgba(255,255,255,0.5)'
                                            }}
                                        ></div>
                                    </div>
                                </div>

                                <div className="card-arrow">
                                    <ArrowRight size={20} color="white" />
                                </div>
                            </Link>
                        </motion.div>
                    );
                })}
            </motion.div>
        </div>
    );
};

export default Home;
