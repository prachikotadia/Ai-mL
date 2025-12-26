import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import FlashcardView from './pages/FlashcardView';
import FavoritesView from './pages/FavoritesView';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/section/:sectionId" element={<FlashcardView />} />
        <Route path="/favorites" element={<FavoritesView />} />
      </Routes>
    </Router>
  );
}

export default App;
