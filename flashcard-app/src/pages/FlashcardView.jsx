import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { ChevronLeft, ChevronRight, RotateCcw, CheckCircle, Home } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Flashcard from '../components/Flashcard';
import cardsDataHelper from '../data/cards.json';
import { getSectionProgress, saveProgress } from '../utils/storage';
import '../App.css';

const FlashcardView = () => {
    const { sectionId } = useParams();
    const navigate = useNavigate();

    // Find section data
    const section = cardsDataHelper.find(s => s.id === sectionId);

    const [currentIndex, setCurrentIndex] = useState(0);
    const [isFlipped, setIsFlipped] = useState(false);
    const [knownCardIds, setKnownCardIds] = useState([]);
    const [favoriteCardIds, setFavoriteCardIds] = useState([]);

    // Load progress on mount
    useEffect(() => {
        if (section) {
            const progress = getSectionProgress(sectionId);
            setKnownCardIds(progress.knownCardIds || []);
            setFavoriteCardIds(progress.favoriteCardIds || []);
            // Optional: Resume from last index
            if (progress.currentIndex !== undefined && progress.currentIndex < section.cards.length) {
                setCurrentIndex(progress.currentIndex);
            }
        }
    }, [sectionId, section]);

    if (!section) {
        return <div className="p-4">Section not found. <Link to="/">Go Home</Link></div>;
    }

    const cards = section.cards;
    const currentCard = cards[currentIndex];
    const isKnown = knownCardIds.includes(currentCard.id);
    const isFavorite = favoriteCardIds.includes(currentCard.id);

    const handleNext = () => {
        if (currentIndex < cards.length - 1) {
            setIsFlipped(false);
            const nextIndex = currentIndex + 1;
            setCurrentIndex(nextIndex);
            saveProgress(sectionId, { currentIndex: nextIndex });
        }
    };

    const handlePrev = () => {
        if (currentIndex > 0) {
            setIsFlipped(false);
            const prevIndex = currentIndex - 1;
            setCurrentIndex(prevIndex);
            saveProgress(sectionId, { currentIndex: prevIndex });
        }
    };

    const handleRestart = () => {
        setIsFlipped(false);
        setCurrentIndex(0);
        saveProgress(sectionId, { currentIndex: 0 });
    };

    const toggleKnown = (e) => {
        e.stopPropagation();
        const newKnown = isKnown
            ? knownCardIds.filter(id => id !== currentCard.id)
            : [...knownCardIds, currentCard.id];

        setKnownCardIds(newKnown);
        // saveProgress inside helper expects (sectionId, dataObject)
        saveProgress(sectionId, { knownCardIds: newKnown });
    };

    const toggleFavorite = (e) => {
        e.stopPropagation();
        const newFavorites = isFavorite
            ? favoriteCardIds.filter(id => id !== currentCard.id)
            : [...favoriteCardIds, currentCard.id];

        setFavoriteCardIds(newFavorites);
        saveProgress(sectionId, { favoriteCardIds: newFavorites });
    };

    const progressPercentage = ((currentIndex + 1) / cards.length) * 100;

    // Stack Logic: Get next 2 cards
    const cardStack = [0, 1, 2].map(offset => {
        const index = currentIndex + offset;
        if (index >= cards.length) return null;
        return {
            ...cards[index],
            index: index, // Actual index in the deck
            offset: offset // 0 = active, 1 = next, 2 = next+1
        };
    }).filter(Boolean).reverse(); // Reverse so back cards render first

    return (
        <div className="app-container">
            <header className="app-header">
                <Link to="/" className="icon-btn">
                    <Home size={20} />
                </Link>
                <div className="header-title">
                    <span className="section-label">Section</span>
                    <h1>{section.title}</h1>
                </div>
                <div style={{ width: 24 }}></div>
            </header>

            <div className="progress-bar-container">
                <div className="progress-bar" style={{ width: `${progressPercentage}%` }}></div>
            </div>

            <main className="main-content" style={{ position: 'relative', height: 'min(540px, 62vh)', marginTop: '20px' }}>
                <AnimatePresence mode="popLayout">
                    {cardStack.map((card) => {
                        const isCurrent = card.index === currentIndex;

                        // Calculated Styles for Stack
                        const stackStyle = {
                            zIndex: 100 - card.offset,
                            transform: `scale(${1 - (card.offset * 0.05)}) translateY(${card.offset * 12}px)`,
                            opacity: 1 - (card.offset * 0.3),
                            filter: card.offset > 0 ? 'brightness(0.9)' : 'none',
                        };

                        return (
                            <motion.div
                                key={card.id}
                                // Layout & Stacking Props
                                style={{
                                    position: 'absolute',
                                    width: '100%',
                                    height: '100%',
                                    top: 0,
                                    left: 0,
                                    zIndex: 100 - card.offset,
                                    // Enable pointer events ONLY if it's the current card
                                    pointerEvents: isCurrent ? 'auto' : 'none'
                                }}

                                // Initial & Exit Animations
                                initial={isCurrent ? { x: 0, scale: 1, opacity: 1 } : { scale: 0.9, opacity: 0 }}
                                animate={{
                                    x: 0,
                                    opacity: 1 - (card.offset * 0.3),
                                    scale: 1 - (card.offset * 0.05),
                                    y: card.offset * 12,
                                    rotate: 0
                                }}
                                exit={(custom) => {
                                    // Custom logic for exit direction
                                    const x = custom === 'left' ? -300 : custom === 'right' ? 300 : -300;
                                    const rotate = custom === 'left' ? -20 : custom === 'right' ? 20 : 0;
                                    return { x, opacity: 0, scale: 0.8, rotate };
                                }}

                                // Gesture Props (Only active for top card)
                                drag={isCurrent ? "x" : false}
                                dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
                                dragElastic={0.6}
                                whileTap={isCurrent ? { cursor: 'grabbing', scale: 1.02 } : {}}
                                onDragEnd={(e, { offset, velocity }) => {
                                    if (!isCurrent) return;

                                    const swipeThreshold = 50;
                                    if (offset.x < -swipeThreshold) {
                                        // Swipe Left -> Next
                                        handleNext();
                                    } else if (offset.x > swipeThreshold) {
                                        // Swipe Right -> Prev
                                        handlePrev();
                                    }
                                }}
                            >
                                <Flashcard
                                    data={card}
                                    isFlipped={isCurrent ? isFlipped : false}
                                    onFlip={isCurrent ? () => setIsFlipped(!isFlipped) : undefined}
                                    colorIndex={(parseInt(sectionId.replace('section-', '')) || 0) + card.index}
                                    isFavorite={isCurrent ? isFavorite : (favoriteCardIds.includes(card.id))}
                                    onToggleFavorite={isCurrent ? toggleFavorite : undefined}
                                    style={{}}
                                />
                            </motion.div>
                        );
                    })}
                </AnimatePresence>
            </main>

            {/* Controls */}
            <div className="card-counter" style={{ marginTop: '0px', marginBottom: '10px' }}>
                Card {currentIndex + 1} of {cards.length}
            </div>

            <footer className="footer-controls">
                <div className="control-row">
                    <motion.button
                        whileTap={{ scale: 0.9 }}
                        onClick={toggleKnown}
                        className={`action-btn ${isKnown ? 'known' : ''}`}
                        title="Mark as Known"
                        animate={{
                            backgroundColor: isKnown ? '#dcfce7' : 'rgba(255,255,255,0.1)',
                            color: isKnown ? '#166534' : '#94a3b8'
                        }}
                    >
                        <motion.div
                            initial={false}
                            animate={{ scale: isKnown ? [1, 1.2, 1] : 1 }}
                            transition={{ duration: 0.3 }}
                        >
                            <CheckCircle size={24} />
                        </motion.div>
                        <span>{isKnown ? "Known" : "Mark Known"}</span>
                    </motion.button>
                </div>

                <div className="nav-row">
                    {/* Restart Button */}
                    <motion.button
                        whileTap={{ scale: 0.9 }}
                        className="nav-btn secondary"
                        onClick={handleRestart}
                        title="Restart Section"
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
                            disabled={currentIndex === cards.length - 1}
                            style={{
                                background: 'white',
                                color: '#1e293b',
                                border: 'none',
                                opacity: currentIndex === cards.length - 1 ? 0.5 : 1
                            }}
                        >
                            <ChevronRight size={24} />
                        </motion.button>
                    </div>
                </div>
            </footer>
        </div>
    );
};

export default FlashcardView;
