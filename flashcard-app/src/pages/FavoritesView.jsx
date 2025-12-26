import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ChevronLeft, ChevronRight, RotateCcw, Home, Star } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Flashcard from '../components/Flashcard';
import cardsData from '../data/cards.json';
import { getProgress, saveProgress } from '../utils/storage';
import '../App.css';

const FavoritesView = () => {
    const navigate = useNavigate();
    const [currentIndex, setCurrentIndex] = useState(0);
    const [isFlipped, setIsFlipped] = useState(false);
    const [favoriteCards, setFavoriteCards] = useState([]);

    // We need to reload favorites if they change
    const [reloadTrigger, setReloadTrigger] = useState(0);

    useEffect(() => {
        const allProgress = getProgress();
        const aggregated = [];

        cardsData.forEach(section => {
            const sectionProgress = allProgress[section.id] || {};
            const favIds = sectionProgress.favoriteCardIds || [];

            section.cards.forEach((card, index) => {
                if (favIds.includes(card.id)) {
                    aggregated.push({
                        ...card,
                        originalIndex: index,
                        sectionId: section.id,
                        colorIndex: (parseInt(section.id.replace('section-', '')) || 0) + index
                    });
                }
            });
        });

        setFavoriteCards(aggregated);
    }, [reloadTrigger]);

    const currentCard = favoriteCards[currentIndex];

    const handleNext = () => {
        if (currentIndex < favoriteCards.length - 1) {
            setIsFlipped(false);
            setCurrentIndex(prev => prev + 1);
        }
    };

    const handlePrev = () => {
        if (currentIndex > 0) {
            setIsFlipped(false);
            setCurrentIndex(prev => prev - 1);
        }
    };

    const handleRestart = () => {
        setIsFlipped(false);
        setCurrentIndex(0);
    };

    const toggleFavorite = (e) => {
        e.stopPropagation();
        // Remove from favorites logic
        if (!currentCard) return;

        // We are "Unstarring" basically, since this view ONLY shows stars
        const sectionId = currentCard.sectionId;
        const allProgress = getProgress();
        const sectionProgress = allProgress[sectionId] || {};
        const favIds = sectionProgress.favoriteCardIds || [];

        const newFavIds = favIds.filter(id => id !== currentCard.id);

        saveProgress(sectionId, { favoriteCardIds: newFavIds });

        // Trigger reload to remove card from deck (or just visually update it?)
        // Better UX: keep it visible but un-filled, until they leave? 
        // Or remove immediately? immediate removal might be jarring if it's the current card.
        // Let's force reload so the list updates.
        setReloadTrigger(prev => prev + 1);

        // Adjustment if we remove the last card
        if (currentIndex >= favoriteCards.length - 1 && currentIndex > 0) {
            setCurrentIndex(prev => prev - 1);
        }
    };

    // Stack Logic
    // If no favorites, show empty state
    if (favoriteCards.length === 0) {
        return (
            <div className="app-container" style={{ justifyContent: 'center', alignItems: 'center' }}>
                <header className="app-header">
                    <Link to="/" className="icon-btn">
                        <Home size={20} />
                    </Link>
                    <div className="header-title">
                        <h1>Favorites</h1>
                    </div>
                    <div style={{ width: 24 }}></div>
                </header>
                <div style={{ textAlign: 'center', padding: 40, color: '#94a3b8' }}>
                    <Star size={48} style={{ marginBottom: 16, opacity: 0.5 }} />
                    <h2>No Favorites Yet</h2>
                    <p>Star cards in any section to appear here.</p>
                    <Link to="/" style={{ marginTop: 24, display: 'inline-block', color: 'white', textDecoration: 'underline' }}>
                        Browse Sections
                    </Link>
                </div>
            </div>
        );
    }

    const cardStack = [0, 1, 2].map(offset => {
        const index = currentIndex + offset;
        if (index >= favoriteCards.length) return null;
        return {
            ...favoriteCards[index],
            index: index, // Local index in fav deck
            offset: offset
        };
    }).filter(Boolean).reverse();

    const progressPercentage = ((currentIndex + 1) / favoriteCards.length) * 100;

    return (
        <div className="app-container">
            <header className="app-header">
                <Link to="/" className="icon-btn">
                    <Home size={20} />
                </Link>
                <div className="header-title">
                    <span className="section-label">Review</span>
                    <h1>Favorites</h1>
                </div>
                <div style={{ width: 24 }}></div>
            </header>

            <div className="progress-bar-container">
                <div className="progress-bar" style={{ width: `${progressPercentage}%`, background: '#fbbf24' }}></div>
            </div>

            <main className="main-content" style={{ position: 'relative', height: 'min(540px, 62vh)', marginTop: '20px' }}>
                <AnimatePresence mode="popLayout">
                    {cardStack.map((card) => {
                        const isCurrent = card.index === currentIndex;
                        return (
                            <motion.div
                                key={card.id}
                                style={{
                                    position: 'absolute',
                                    width: '100%',
                                    height: '100%',
                                    top: 0,
                                    left: 0,
                                    zIndex: 100 - card.offset,
                                    pointerEvents: isCurrent ? 'auto' : 'none'
                                }}
                                initial={isCurrent ? { x: 0, scale: 1, opacity: 1 } : { scale: 0.9, opacity: 0 }}
                                animate={{
                                    x: 0,
                                    opacity: 1 - (card.offset * 0.3),
                                    scale: 1 - (card.offset * 0.05),
                                    y: card.offset * 12,
                                    rotate: 0
                                }}
                                exit={{ x: -300, opacity: 0 }}
                                drag={isCurrent ? "x" : false}
                                dragConstraints={{ left: 0, right: 0 }}
                                onDragEnd={(e, { offset }) => {
                                    if (!isCurrent) return;
                                    if (offset.x < -50) handleNext();
                                    else if (offset.x > 50) handlePrev();
                                }}
                            >
                                <Flashcard
                                    data={card}
                                    isFlipped={isCurrent ? isFlipped : false}
                                    onFlip={isCurrent ? () => setIsFlipped(!isFlipped) : undefined}
                                    colorIndex={card.colorIndex}
                                    isFavorite={true}
                                    onToggleFavorite={isCurrent ? toggleFavorite : undefined}
                                    style={{}}
                                />
                            </motion.div>
                        );
                    })}
                </AnimatePresence>
            </main>

            <div className="card-counter" style={{ marginTop: '0px', marginBottom: '10px' }}>
                Card {currentIndex + 1} of {favoriteCards.length}
            </div>

            <footer className="footer-controls">
                <div className="control-row" style={{ justifyContent: 'center' }}>
                    <div className="nav-row">
                        <motion.button
                            whileTap={{ scale: 0.9 }}
                            className="nav-btn secondary"
                            onClick={handleRestart}
                            style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', color: '#94a3b8' }}
                        >
                            <RotateCcw size={20} />
                        </motion.button>
                        <div style={{ display: 'flex', gap: '16px' }}>
                            <motion.button
                                whileTap={{ scale: 0.9 }}
                                className="nav-btn primary"
                                onClick={handlePrev}
                                disabled={currentIndex === 0}
                                style={{ opacity: currentIndex === 0 ? 0.5 : 1 }}
                            >
                                <ChevronLeft size={24} />
                            </motion.button>
                            <motion.button
                                whileTap={{ scale: 0.9 }}
                                className="nav-btn primary"
                                onClick={handleNext}
                                disabled={currentIndex === favoriteCards.length - 1}
                                style={{
                                    background: 'white',
                                    color: '#1e293b',
                                    border: 'none',
                                    opacity: currentIndex === favoriteCards.length - 1 ? 0.5 : 1
                                }}
                            >
                                <ChevronRight size={24} />
                            </motion.button>
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    );
};

export default FavoritesView;
