%   The algorithms implemented by Alexander Vezhnevets aka Vezhnick
%   <a>href="mailto:vezhnick@gmail.com">vezhnick@gmail.com</a>
%
%   Copyright (C) 2005, Vezhnevets Alexander
%   vezhnick@gmail.com
%   
%   This file is part of GML Matlab Toolbox
%   For conditions of distribution and use, see the accompanying License.txt file.
%
%   Initialize initializes crossvalidation object with data
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%
%    this = Initialize(this, Data, Labels)
%    ---------------------------------------------------------------------------------
%    Arguments:
%           this     - crossvalidation object
%           Data     - Should be DxN matrix, where D is the
%                      dimensionality of data, and N is the number of
%                      samples.
%           Labels   - Should be 1xN matrix, where N is
%                      the number of samples.
%    Return:
%           this     - crossvalidation object, initialized with data and
%                      labels

function this = Initialize(this, Data, Labels)

if( size(Labels,1) ~= 1 || length(size(Labels)) ~= 2)
  error('Labels should be a (1,N) matrix.');
end

if( size(Labels,2) ~= size(Data,2))
  error('Data should be (M,N) matrix and Labels (1,N)');
end

for i = 1 : length(Data)
    fold_i = floor(rand * (this.folds - eps) + 1);
%     fold_i = mod(i,this.folds) + 1;
    if( i > this.folds )
        this.CrossDataSets{fold_i} = {cat(2, Data(:, i), this.CrossDataSets{fold_i}{1,1})};
        this.CrossLabelsSets{fold_i} = {cat(2, Labels(:, i), this.CrossLabelsSets{fold_i}{1,1})};
    else
        this.CrossDataSets{i} = {cat(2, Data(:, i), this.CrossDataSets{i})};
        this.CrossLabelsSets{i} = {cat(2, Labels(:, i), this.CrossLabelsSets{i})};
    end
%     if( i > this.folds )
%         this.CrossDataSets{mod(i,this.folds) + 1} = {cat(2, Data(:, i), this.CrossDataSets{mod(i,this.folds) + 1}{1,1})};
%         this.CrossLabelsSets{mod(i,this.folds) + 1} = {cat(2, Labels(:, i), this.CrossLabelsSets{mod(i,this.folds) + 1}{1,1})};
%     else
%         this.CrossDataSets{mod(i,this.folds) + 1} = {cat(2, Data(:, i), this.CrossDataSets{mod(i,this.folds) + 1})};
%         this.CrossLabelsSets{mod(i,this.folds) + 1} = {cat(2, Labels(:, i), this.CrossLabelsSets{mod(i,this.folds) + 1})};
%     end
end
