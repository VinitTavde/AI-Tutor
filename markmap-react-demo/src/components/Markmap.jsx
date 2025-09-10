import React, { useRef, useEffect } from 'react';
import { Transformer } from 'markmap-lib';
import { Markmap, loadCSS, loadJS } from 'markmap-view';

const transformer = new Transformer();

const MarkmapComponent = ({ markdown }) => {
  const ref = useRef();
  const markmapRef = useRef();

  useEffect(() => {
    if (markdown) {
      const { root } = transformer.transform(markdown);
      if (markmapRef.current) {
        markmapRef.current.setData(root);
      } else {
        markmapRef.current = Markmap.create(ref.current, {
          embedGlobalCSS: true,
          fitView: true,
          pan: true,
          zoom: true,
        }, root);
      }
      markmapRef.current.fit(); // Fit the view after data is set

      const { styles, scripts } = transformer.getAssets();
      loadCSS(styles);
      loadJS(scripts);
    }
  }, [markdown]);

  const handleDoubleClick = () => {
    if (markmapRef.current) {
      markmapRef.current.fit();
    }
  };

  return <svg ref={ref} style={{ width: '100%', height: '100vh' }} onDoubleClick={handleDoubleClick} />;
};

export default MarkmapComponent; 