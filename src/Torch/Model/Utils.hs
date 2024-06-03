module Torch.Model.Utils (
    accuracy,
    precision,
    recall,
    f1,
    macroAvg,
    weightedAvg
  ) where

import Prelude hiding (add, mul, div)
import Torch.Functional (add, mul, div, sumAll)
import Torch.Tensor (Tensor(..), asTensor, asValue)
import Torch.TensorFactories (zeros')
import Torch.Tensor.TensorFactories (oneAtPos2d)
import Torch.Tensor.Util (indexOfMax,oneHot')
import Torch.Layer.MLP      (MLPParams(..))

import Graphics.Matplotlib (Matplotlib(..), o2, title, xlabel, ylabel, colorbar, mp, setSizeInches, pcolor, text, (%), (@@), (#))

-- | Return the accuracy of a model
-- * model : The model
-- * forward : The forward function for the model
-- * dataSet : The dataset
accuracy :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Float
accuracy model forward dataSet 
    | null dataSet = 0.0  -- If the dataset is empty, return 0 as default accuracy
    | otherwise = (sum results) / (fromIntegral $ length results)
    where results = map (\(input, output) -> if (indexOfMax $ (asValue (forward model input) :: [Float])) == (indexOfMax $ (asValue output :: [Float])) then 1 else 0) dataSet
       
-- | Return the precision of a model
-- * model : The model
-- * forward : The forward function for the model
-- * dataSet : The dataset
precision :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor, Tensor)] -> Tensor
precision model forward dataSet = tp `div` (tp `add` fp)
    where (tp, _, fp) = getTpFnFp model forward dataSet

-- | Return the recall of a model
-- * model : The model
-- * forward : The forward function for the model
-- * dataSet : The dataset
recall :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Tensor
recall model forward dataSet = tp `div` (tp `add` fn)
    where (tp, fn, _) = getTpFnFp model forward dataSet

-- | Return the f1 score of a model
-- * model : The model
-- * forward : The forward function for the model
-- * dataSet : The dataset
f1 :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Tensor
f1 model forward dataSet = tp `div` (tp `add` ((fn `add` fp) `div` 2.0))
    where (tp, fn, fp) = getTpFnFp model forward dataSet

-- | Return the macro average of the f1 score of a model
-- * model : The model
-- * forward : The forward function for the model
-- * dataSet : The dataset
macroAvg :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Float
macroAvg model forward dataSet = (asValue (sumAll f1score) :: Float) / (fromIntegral $ length (asValue f1score :: [Float]))
    where f1score = f1 model forward dataSet

-- | Return the weighted average of the f1 score of a model
-- * model : The model
-- * forward : The forward function for the model
-- * dataSet : The dataset
weightedAvg :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Float
weightedAvg model forward dataSet = result
    where expecteds = map snd dataSet
          weights = (foldl1 (add) expecteds) `div` (fromIntegral $ length dataSet) 
          f1score = f1 model forward dataSet
          weightedF1 = weights `mul` f1score
          result = asValue (sumAll weightedF1) :: Float


----

-- | Return the true positive, false negative and false positive of one class
getOneTpFnFp :: Tensor -> Tensor -> (Tensor,Tensor,Tensor)
getOneTpFnFp expected guess = result
    where expectedValue = asValue expected :: [Float]
          guessValue = asValue guess :: [Float] 
          nul = zeros' [(length (asValue expected :: [Float]))]
          result = if indexOfMax expectedValue == indexOfMax guessValue then (expected, nul, nul) else (nul, expected, guess)

-- | Return the true positive, false negative and false positive of a classification model
getTpFnFp :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> (Tensor,Tensor,Tensor)
getTpFnFp model forward dataSet = (tp, fn, fp)
    where fTpFnFp = [ getOneTpFnFp output $ oneHot' $ forward model input | (input, output) <- dataSet]
          tp = foldl1 (add) [tps | (tps, _, _) <- fTpFnFp]
          fn = foldl1 (add) [fns | (_, fns, _) <- fTpFnFp]
          fp = foldl1 (add) [fps | (_, _, fps) <- fTpFnFp]
